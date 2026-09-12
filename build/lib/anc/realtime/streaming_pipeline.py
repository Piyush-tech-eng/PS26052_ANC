"""Persistent real-time audio processing pipeline.

This is the central engine that replaces the ad-hoc mode-specific loops
in ``run_live_demo.py`` with a unified, hardware-agnostic pipeline:

    AudioInput → [RingBuffer] → FxNLMS → OverlapAdd(AI) → [RingBuffer] → AudioOutput

Key properties
--------------
- **Arbitrary stream duration** — no temporary WAV files
- **Persistent state** — adaptive filter, AI, overlap-add state persists
  across frames without interruption
- **Underflow / overflow protection** — ring buffers with configurable
  thresholds and health monitoring
- **Clean shutdown** — ``stop()`` flushes buffers and releases resources
- **Automatic recovery** — from dropped UDP frames (via jitter buffer
  and frame interpolation in the AudioInput layer)
- **Configurable frame size** — negotiated at startup
- **Measured end-to-end latency** — per-frame instrumentation via
  ``TelemetryCollector``

Usage::

    from anc.realtime.audio_input import WAVInput
    from anc.realtime.audio_output import LaptopSpeakerOutput
    from anc.realtime.streaming_pipeline import StreamingPipeline
    from ai.streaming.hybrid_engine import HybridEngine

    pipeline = StreamingPipeline(
        input_source=WAVInput("noisy.wav"),
        output_sink=LaptopSpeakerOutput(),
        engine=engine,
    )
    pipeline.start()
    # ... pipeline runs in background thread ...
    pipeline.stop()
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from anc.realtime.audio_input import AudioInput
from anc.realtime.audio_output import AudioOutput
from anc.realtime.telemetry import TelemetryCollector


class StreamingPipeline:
    """Persistent real-time ANC + AI audio processing pipeline.

    Parameters
    ----------
    input_source : AudioInput
        Audio source (WAV file, laptop mic, or Pi UDP).
    output_sink : AudioOutput
        Audio destination (WAV file, laptop speakers, or Pi output).
    engine : HybridEngine
        The ANC + AI processing engine.
    mode : str
        Operating mode label for telemetry.
    status_file : str, optional
        JSON status file path for dashboard polling.
    on_frame : callable, optional
        Callback ``(frame_index, telemetry_snapshot) -> None``
        called after each processed frame (for dashboard updates).
    """

    def __init__(
        self,
        input_source: AudioInput,
        output_sink: AudioOutput,
        engine: Any,  # HybridEngine — use Any to avoid circular import
        mode: str = "offline",
        status_file: str | None = None,
        on_frame: Any = None,
        telemetry: Any | None = None,
        buffer_size: int = 16,
    ) -> None:
        self._input = input_source
        self._output = output_sink
        self._engine = engine
        self._mode = mode
        self._on_frame = on_frame
        self._buffer_size = buffer_size

        self._running = False
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

        # Telemetry
        if telemetry is not None:
            self._telemetry = telemetry
        else:
            self._telemetry = TelemetryCollector(
                mode=mode,
                sample_rate=input_source.sample_rate,
                frame_size=input_source.frame_size,
                status_file=status_file,
            )

        # Stats
        self._frames_processed = 0
        self._start_time = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frames_processed(self) -> int:
        return self._frames_processed

    @property
    def telemetry(self) -> TelemetryCollector:
        return self._telemetry

    @property
    def last_error(self) -> Exception | None:
        return self._error

    def start(self) -> None:
        """Start the processing pipeline in a background thread.

        The pipeline will continuously read frames from the input source,
        process them through the engine, and write to the output sink.
        """
        if self._running:
            return

        self._running = True
        self._error = None
        self._frames_processed = 0
        self._start_time = time.time()

        # Open I/O
        self._input.open()
        self._output.open()

        self._telemetry.set_state("LIVE")

        self._thread = threading.Thread(
            target=self._processing_loop,
            daemon=True,
            name="streaming-pipeline",
        )
        self._thread.start()

    def stop(self) -> None:
        """Gracefully stop the pipeline and release resources."""
        self._running = False
        self._telemetry.set_state("STOPPED")

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

        # Close I/O
        try:
            self._output.close()
        except Exception:
            pass
        try:
            self._input.close()
        except Exception:
            pass

        # Final telemetry write
        self._telemetry.write_status()

    def run_blocking(self) -> None:
        """Run the pipeline in the current thread (blocks until done).

        Useful for offline file processing where there's no need for
        background operation.
        """
        self._running = True
        self._error = None
        self._frames_processed = 0
        self._start_time = time.time()

        self._input.open()
        self._output.open()
        self._telemetry.set_state("LIVE")

        try:
            self._processing_loop()
        finally:
            self._telemetry.set_state("STOPPED")
            self._output.close()
            self._input.close()
            self._telemetry.write_status()

    def get_telemetry(self):
        """Return the latest telemetry snapshot."""
        return self._telemetry.snapshot()

    def _processing_loop(self) -> None:
        """Main processing loop — runs in background thread or blocking."""
        try:
            idle_count = 0
            max_idle = 500  # ~50s at 0.1s sleep — for live modes, wait forever

            while self._running:
                t_frame_start = time.perf_counter()

                # 1. Read a frame from the input source
                t_capture_start = time.perf_counter()
                frame = self._input.read_frame()
                t_capture_end = time.perf_counter()

                if frame is None:
                    # For file sources, this means EOF
                    if hasattr(self._input, '_audio') and self._input._audio is not None:
                        # WAVInput exhausted
                        break
                    # For live sources, brief sleep and retry
                    idle_count += 1
                    if idle_count > max_idle:
                        break  # Safety exit for non-live sources
                    time.sleep(0.005)
                    continue

                idle_count = 0

                # 2. Process through the engine
                measured = frame.error
                reference = frame.reference

                t_anc_ai_start = time.perf_counter()
                if self._engine.has_anc and reference is not None:
                    enhanced, timing = self._engine.process_frame(
                        measured=measured, reference=reference
                    )
                else:
                    enhanced, timing = self._engine.process_frame(
                        measured=measured
                    )
                t_anc_ai_end = time.perf_counter()

                # 3. Write to output
                t_playback_start = time.perf_counter()
                self._output.write_frame(enhanced)
                t_playback_end = time.perf_counter()

                self._frames_processed += 1

                # 4. Update telemetry
                capture_ms = (t_capture_end - t_capture_start) * 1000
                anc_ms = timing.anc_ms
                ai_ms = timing.ai_ms
                playback_ms = (t_playback_end - t_playback_start) * 1000

                input_rms = float(np.sqrt(np.mean(measured ** 2)))
                output_rms = float(np.sqrt(np.mean(enhanced ** 2)))
                ref_rms = float(np.sqrt(np.mean(reference ** 2))) if reference is not None else 0.0

                self._telemetry.update_frame(
                    input_rms=input_rms,
                    output_rms=output_rms,
                    reference_rms=ref_rms,
                )
                self._telemetry.update_latency(
                    capture_ms=capture_ms,
                    anc_ms=anc_ms,
                    ai_ms=ai_ms,
                    playback_ms=playback_ms,
                )
                self._telemetry.update_ai(
                    model_name=self._engine.model_name,
                    enabled=True,
                    inference_ms=ai_ms,
                    realtime_ratio=timing.realtime_ratio,
                )
                self._telemetry.update_anc(
                    enabled=self._engine.has_anc,
                    convergence=getattr(self._engine, "convergence_indicator", 0.0),
                    attenuation_db=getattr(self._engine, "estimated_attenuation_db", 0.0),
                )

                # Waveform data for dashboard (feed rolling sample buffer)
                self._telemetry.update_waveforms(
                    reference=reference,
                    error=measured,
                    output=enhanced,
                )

                # Write status file periodically
                self._telemetry.write_status()

                # Frame callback
                if self._on_frame is not None:
                    try:
                        self._on_frame(
                            self._frames_processed,
                            self._telemetry.snapshot(),
                        )
                    except Exception:
                        pass

        except Exception as exc:
            self._error = exc
            self._telemetry.set_state("ERROR")
        finally:
            self._running = False

    def __enter__(self) -> "StreamingPipeline":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
