"""Tests for continuous streaming audio pipeline and hardware abstraction."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile

from anc.realtime.ring_buffer import AudioRingBuffer
from anc.realtime.audio_input import AudioFrame, WAVInput
from anc.realtime.audio_output import WAVOutput
from anc.realtime.telemetry import TelemetryCollector, PipelineTelemetry
from anc.realtime.streaming_pipeline import StreamingPipeline
from ai.models.base import EnhancementModel


class DummyMockModel(EnhancementModel):
    """Mock model that returns identical audio length."""

    @property
    def name(self) -> str:
        return "mock_model"

    @property
    def sample_rate(self) -> int:
        return 16_000

    @property
    def frame_size(self) -> int:
        return 128

    def enhance(self, x: np.ndarray) -> np.ndarray:
        return x * 0.5


class DummyEngine:
    """Mock hybrid engine for pipeline unit testing."""

    def __init__(self, sample_rate: int = 16_000) -> None:
        self.sample_rate = sample_rate
        self.has_anc = True
        self.model_name = "mock_model"
        self.convergence_indicator = 0.85
        self.estimated_attenuation_db = -12.4

    def process_frame(self, measured: np.ndarray, reference: np.ndarray | None = None):
        from ai.streaming.hybrid_engine import StageTimings
        enhanced = measured * 0.5
        timing = StageTimings(
            anc_seconds=0.001,
            ai_seconds=0.003,
            total_seconds=0.004,
            frame_samples=len(measured),
            sample_rate=self.sample_rate,
        )
        return enhanced, timing


class TestAudioRingBuffer:
    """Unit tests for thread-safe circular audio ring buffer."""

    def test_write_read_basic(self) -> None:
        rb = AudioRingBuffer(capacity_samples=1000, channels=2)
        assert rb.fill_level() == 0
        assert rb.fill_ratio() == 0.0

        ref = np.ones(320)
        err = np.zeros(320)
        written = rb.write(ref, err)
        assert written == 320
        assert rb.fill_level() == 320

        out_ref, out_err = rb.read(320)
        assert np.array_equal(out_ref, ref)
        assert np.array_equal(out_err, err)
        assert rb.fill_level() == 0

    def test_overflow_and_underflow_stats(self) -> None:
        rb = AudioRingBuffer(capacity_samples=500, channels=1, overflow_policy="drop_oldest")
        # Underflow read returns None
        read_empty = rb.read(100)
        assert read_empty is None
        assert rb.health().underflows == 1

        # Overflow write
        rb.write(np.ones(600))
        assert rb.health().overflows > 0


class TestWAVInputOutputRoundtrip:
    """Unit tests for WAVInput and WAVOutput streaming file interfaces."""

    def test_stereo_wav_roundtrip(self, tmp_path: Path) -> None:
        sr = 16_000
        duration_s = 0.5
        n = int(sr * duration_s)

        # Create synthetic 2-channel WAV (ch0=error, ch1=reference)
        ch0 = (0.3 * np.sin(2 * np.pi * 440 * np.arange(n) / sr) * 32767).astype(np.int16)
        ch1 = (0.5 * np.sin(2 * np.pi * 200 * np.arange(n) / sr) * 32767).astype(np.int16)
        stereo = np.column_stack([ch0, ch1])

        in_file = tmp_path / "test_stereo.wav"
        out_file = tmp_path / "test_out.wav"
        wavfile.write(str(in_file), sr, stereo)

        wav_in = WAVInput(in_file, frame_ms=20, target_sample_rate=sr, reference_channel=1, error_channel=0)
        wav_out = WAVOutput(out_file, sample_rate=sr)

        wav_in.open()
        wav_out.open()

        frames_read = 0
        while True:
            f = wav_in.read_frame()
            if f is None:
                break
            assert f.error is not None
            assert f.reference is not None
            assert len(f.error) == 320
            assert len(f.reference) == 320
            wav_out.write_frame(f.error * 0.5)
            frames_read += 1

        wav_out.close()
        wav_in.close()

        assert frames_read == 25  # 0.5s / 0.02s = 25 frames
        assert out_file.exists()

        sr_out, out_data = wavfile.read(str(out_file))
        assert sr_out == sr
        assert len(out_data) == n


class TestStreamingPipeline:
    """Unit tests for the StreamingPipeline coordinator."""

    def test_pipeline_execution(self, tmp_path: Path) -> None:
        sr = 16_000
        n = int(sr * 0.2)  # 10 frames
        t = np.arange(n) / sr
        audio = (0.4 * np.sin(2 * np.pi * 300 * t) * 32767).astype(np.int16)

        in_file = tmp_path / "pipe_in.wav"
        out_file = tmp_path / "pipe_out.wav"
        status_file = tmp_path / "status.json"
        wavfile.write(str(in_file), sr, audio)

        wav_in = WAVInput(in_file, frame_ms=20, target_sample_rate=sr)
        wav_out = WAVOutput(out_file, sample_rate=sr)
        engine = DummyEngine(sample_rate=sr)

        pipeline = StreamingPipeline(
            input_source=wav_in,
            output_sink=wav_out,
            engine=engine,
            mode="offline",
            status_file=str(status_file),
        )

        pipeline.run_blocking()

        assert pipeline.frames_processed == 10
        assert out_file.exists()
        assert status_file.exists()

        telemetry = pipeline.get_telemetry()
        assert telemetry.state == "STOPPED"
        assert telemetry.frames_processed == 10
        assert telemetry.sample_rate == sr
