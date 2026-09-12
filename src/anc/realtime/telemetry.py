"""Real-time pipeline telemetry for PS26052 ANC.

Provides :class:`PipelineTelemetry` — a comprehensive snapshot of the
running system state — and :class:`TelemetryCollector` which accumulates
per-frame measurements into rolling statistics.

The telemetry is consumed by:
- The live dashboard (via WebSocket or JSON polling)
- The status JSON file
- Console logging during demos
"""

from __future__ import annotations

import collections
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class PipelineTelemetry:
    """Complete snapshot of the running ANC system state.

    All timing values are in milliseconds unless otherwise noted.
    """

    # ── System state ─────────────────────────────────────────────────
    state: str = "STOPPED"          # LIVE, STOPPED, CALIBRATING, ERROR
    mode: str = "offline"           # offline, laptop_live, pi_prototype, pi_edge
    sample_rate: int = 16_000
    frame_size: int = 320
    frames_processed: int = 0
    elapsed_seconds: float = 0.0
    timestamp: float = 0.0

    # ── System resources ─────────────────────────────────────────────
    cpu_percent: float = 0.0
    ram_mb: float = 0.0

    # ── Audio levels ─────────────────────────────────────────────────
    input_rms: float = 0.0
    output_rms: float = 0.0
    reference_rms: float = 0.0

    # ── Buffer health ────────────────────────────────────────────────
    input_buffer_fill: float = 0.0   # 0.0–1.0
    output_buffer_fill: float = 0.0  # 0.0–1.0
    buffer_overflows: int = 0
    buffer_underflows: int = 0
    dropped_frames: int = 0

    # ── ANC state ────────────────────────────────────────────────────
    anc_enabled: bool = False
    filter_length: int = 64
    step_size: float = 0.01
    convergence_indicator: float = 0.0   # 0.0–1.0
    estimated_attenuation_db: float = 0.0

    # ── AI state ─────────────────────────────────────────────────────
    model_name: str = "none"
    ai_enabled: bool = False
    model_precision: str = "fp32"
    inference_time_ms: float = 0.0
    realtime_ratio: float = 0.0

    # ── Latency breakdown ────────────────────────────────────────────
    capture_latency_ms: float = 0.0
    transport_latency_ms: float = 0.0
    anc_latency_ms: float = 0.0
    ai_latency_ms: float = 0.0
    playback_latency_ms: float = 0.0
    total_latency_ms: float = 0.0

    # ── Waveform data (for dashboard) ────────────────────────────────
    # These are not serialized to the JSON status file, only via WebSocket
    reference_waveform: list[float] = field(default_factory=list)
    error_waveform: list[float] = field(default_factory=list)
    output_waveform: list[float] = field(default_factory=list)

    def to_dict(self, include_waveforms: bool = False) -> dict[str, Any]:
        """Serialize to JSON-compatible dict.

        Parameters
        ----------
        include_waveforms : bool
            If True, include raw waveform arrays (for WebSocket).
            Default False (for JSON status file — too large).
        """
        d = asdict(self)
        if not include_waveforms:
            d.pop("reference_waveform", None)
            d.pop("error_waveform", None)
            d.pop("output_waveform", None)
        return d

    def to_json(self, include_waveforms: bool = False) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(include_waveforms), indent=2)


class TelemetryCollector:
    """Accumulates per-frame measurements and produces telemetry snapshots.

    Thread-safe: multiple producers can call ``update_*`` methods while a
    consumer reads ``snapshot()``.

    Parameters
    ----------
    mode : str
        Operating mode identifier.
    sample_rate : int
        Audio sample rate.
    frame_size : int
        Samples per processing frame.
    status_file : str or Path, optional
        Path to write JSON telemetry for dashboard polling.
    write_interval : float
        Minimum seconds between status file writes (default 0.1).
    """

    def __init__(
        self,
        mode: str = "offline",
        sample_rate: int = 16_000,
        frame_size: int = 320,
        status_file: str | Path | None = None,
        write_interval: float = 0.1,
    ) -> None:
        self._mode = mode
        self._sample_rate = sample_rate
        self._frame_size = frame_size
        self._status_file = Path(status_file) if status_file else None
        self._write_interval = write_interval

        self._start_time = time.time()
        self._frames = 0
        self._last_write = 0.0

        # Latest per-frame values (updated atomically via assignment)
        self._input_rms = 0.0
        self._output_rms = 0.0
        self._reference_rms = 0.0

        self._anc_enabled = False
        self._filter_length = 64
        self._step_size = 0.01
        self._convergence = 0.0
        self._attenuation_db = 0.0

        self._model_name = "none"
        self._ai_enabled = False
        self._model_precision = "fp32"
        self._inference_ms = 0.0
        self._rt_ratio = 0.0

        self._capture_ms = 0.0
        self._transport_ms = 0.0
        self._anc_ms = 0.0
        self._ai_ms = 0.0
        self._playback_ms = 0.0
        self._total_ms = 0.0

        self._input_buf_fill = 0.0
        self._output_buf_fill = 0.0
        self._overflows = 0
        self._underflows = 0
        self._dropped = 0

        self._state = "STOPPED"

        # Rolling audio sample buffers (~2048 samples per channel)
        self._buffer_maxlen = 2048
        self._ref_sample_buf: collections.deque[float] = collections.deque(maxlen=self._buffer_maxlen)
        self._err_sample_buf: collections.deque[float] = collections.deque(maxlen=self._buffer_maxlen)
        self._out_sample_buf: collections.deque[float] = collections.deque(maxlen=self._buffer_maxlen)

        # Decimated waveforms for dashboard
        self._ref_waveform: list[float] = []
        self._err_waveform: list[float] = []
        self._out_waveform: list[float] = []

    def set_state(self, state: str) -> None:
        self._state = state

    def update_frame(
        self,
        *,
        input_rms: float = 0.0,
        output_rms: float = 0.0,
        reference_rms: float = 0.0,
    ) -> None:
        """Record per-frame audio level measurements."""
        self._frames += 1
        self._input_rms = input_rms
        self._output_rms = output_rms
        self._reference_rms = reference_rms

    def update_anc(
        self,
        *,
        enabled: bool = True,
        filter_length: int = 64,
        step_size: float = 0.01,
        convergence: float = 0.0,
        attenuation_db: float = 0.0,
    ) -> None:
        """Update ANC subsystem state."""
        self._anc_enabled = enabled
        self._filter_length = filter_length
        self._step_size = step_size
        self._convergence = convergence
        self._attenuation_db = attenuation_db

    def update_ai(
        self,
        *,
        model_name: str = "none",
        enabled: bool = True,
        precision: str = "fp32",
        inference_ms: float = 0.0,
        realtime_ratio: float = 0.0,
    ) -> None:
        """Update AI subsystem state."""
        self._model_name = model_name
        self._ai_enabled = enabled
        self._model_precision = precision
        self._inference_ms = inference_ms
        self._rt_ratio = realtime_ratio

    def set_model_precision(self, precision: str) -> None:
        """Set the active AI model precision tier."""
        self._model_precision = precision

    def update_latency(
        self,
        *,
        capture_ms: float = 0.0,
        transport_ms: float = 0.0,
        anc_ms: float = 0.0,
        ai_ms: float = 0.0,
        playback_ms: float = 0.0,
    ) -> None:
        """Update per-stage latency measurements."""
        self._capture_ms = capture_ms
        self._transport_ms = transport_ms
        self._anc_ms = anc_ms
        self._ai_ms = ai_ms
        self._playback_ms = playback_ms
        self._total_ms = capture_ms + transport_ms + anc_ms + ai_ms + playback_ms

    def update_buffers(
        self,
        *,
        input_fill: float = 0.0,
        output_fill: float = 0.0,
        overflows: int = 0,
        underflows: int = 0,
        dropped: int = 0,
    ) -> None:
        """Update buffer health metrics."""
        self._input_buf_fill = input_fill
        self._output_buf_fill = output_fill
        self._overflows = overflows
        self._underflows = underflows
        self._dropped = dropped

    def update_waveforms(
        self,
        reference: Any | None = None,
        error: Any | None = None,
        output: Any | None = None,
        max_points: int = 256,
    ) -> None:
        """Store real audio samples into rolling buffers and decimate for dashboard."""
        import numpy as np

        def _decimate_buf(buf: collections.deque[float], n: int) -> list[float]:
            if not buf:
                return []
            arr = list(buf)
            if len(arr) <= n:
                return [round(float(v), 4) for v in arr]
            indices = np.linspace(0, len(arr) - 1, n, dtype=int)
            return [round(float(arr[i]), 4) for i in indices]

        if reference is not None:
            ref_flat = np.asarray(reference, dtype=np.float32).ravel()
            self._ref_sample_buf.extend(ref_flat.tolist())
            self._ref_waveform = _decimate_buf(self._ref_sample_buf, max_points)
        if error is not None:
            err_flat = np.asarray(error, dtype=np.float32).ravel()
            self._err_sample_buf.extend(err_flat.tolist())
            self._err_waveform = _decimate_buf(self._err_sample_buf, max_points)
        if output is not None:
            out_flat = np.asarray(output, dtype=np.float32).ravel()
            self._out_sample_buf.extend(out_flat.tolist())
            self._out_waveform = _decimate_buf(self._out_sample_buf, max_points)

    def snapshot(self) -> PipelineTelemetry:
        """Build and return a complete telemetry snapshot."""
        now = time.time()

        # Try to read CPU/RAM (best-effort, no hard dependency)
        cpu_pct = 0.0
        ram_mb = 0.0
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=None)
            proc = psutil.Process(os.getpid())
            ram_mb = proc.memory_info().rss / (1024 * 1024)
        except (ImportError, Exception):
            pass

        return PipelineTelemetry(
            state=self._state,
            mode=self._mode,
            sample_rate=self._sample_rate,
            frame_size=self._frame_size,
            frames_processed=self._frames,
            elapsed_seconds=now - self._start_time,
            timestamp=now,
            cpu_percent=cpu_pct,
            ram_mb=ram_mb,
            input_rms=self._input_rms,
            output_rms=self._output_rms,
            reference_rms=self._reference_rms,
            input_buffer_fill=self._input_buf_fill,
            output_buffer_fill=self._output_buf_fill,
            buffer_overflows=self._overflows,
            buffer_underflows=self._underflows,
            dropped_frames=self._dropped,
            anc_enabled=self._anc_enabled,
            filter_length=self._filter_length,
            step_size=self._step_size,
            convergence_indicator=self._convergence,
            estimated_attenuation_db=self._attenuation_db,
            model_name=self._model_name,
            ai_enabled=self._ai_enabled,
            model_precision=self._model_precision,
            inference_time_ms=self._inference_ms,
            realtime_ratio=self._rt_ratio,
            capture_latency_ms=self._capture_ms,
            transport_latency_ms=self._transport_ms,
            anc_latency_ms=self._anc_ms,
            ai_latency_ms=self._ai_ms,
            playback_latency_ms=self._playback_ms,
            total_latency_ms=self._total_ms,
            reference_waveform=self._ref_waveform,
            error_waveform=self._err_waveform,
            output_waveform=self._out_waveform,
        )

    def write_status(self) -> None:
        """Write the current snapshot to the JSON status file.

        Rate-limited to ``write_interval`` seconds.
        """
        if self._status_file is None:
            return
        now = time.time()
        if now - self._last_write < self._write_interval:
            return
        self._last_write = now

        try:
            snap = self.snapshot()
            self._status_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._status_file.with_suffix(".tmp")
            tmp.write_text(snap.to_json(include_waveforms=True) + "\n", encoding="utf-8")
            tmp.replace(self._status_file)
        except Exception:
            pass
