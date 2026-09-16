"""Compatibility and regression test suite for Hybrid Engine.

Verifies:
1. Multi-channel audio decoding (correlation-aware stereo downmix).
2. Parity between Hybrid and AI-only modes on real stereo speech files.
3. Speech crosstalk and cancellation protection in both pipeline and streaming engine.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from anc.interface.processor import AudioProcessingPipeline
from ai.models.spectral_gate import SpectralGateModel
from ai.streaming.hybrid_engine import HybridEngine
from ai.streaming.frame_anc import FrameANCConfig


@pytest.fixture
def pipeline():
    return AudioProcessingPipeline()


class TestHybridCompatibility:
    """Test channel decoding and hybrid speech preservation."""

    def test_decode_stereo_correlated_downmix(self, pipeline: AudioProcessingPipeline):
        """Standard stereo speech audio must downmix to mono with ref_noise=None."""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        speech = 0.4 * np.sin(2 * np.pi * 300 * t)
        stereo = np.column_stack([speech, speech * 0.98 + 0.01 * np.random.randn(len(t))])

        buf = io.BytesIO()
        sf.write(buf, stereo, sr, format="WAV")
        data = buf.getvalue()

        audio_in, noise_ref, decoded_sr = pipeline.decode_audio_bytes(data)
        assert audio_in.ndim == 1
        assert len(audio_in) == len(t)
        assert noise_ref is None
        assert decoded_sr == sr

    def test_decode_dual_mic_uncorrelated_reference(self, pipeline: AudioProcessingPipeline):
        """Uncorrelated channel 1 represents a dedicated acoustic noise reference."""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        speech = 0.4 * np.sin(2 * np.pi * 300 * t)
        noise = 0.2 * np.random.default_rng(42).standard_normal(len(t))
        dual_mic = np.column_stack([speech, noise])

        buf = io.BytesIO()
        sf.write(buf, dual_mic, sr, format="WAV")
        data = buf.getvalue()

        audio_in, noise_ref, decoded_sr = pipeline.decode_audio_bytes(data)
        assert audio_in.ndim == 1
        assert noise_ref is not None
        assert len(noise_ref) == len(t)
        assert decoded_sr == sr

    @pytest.mark.parametrize("wav_filename", ["demo_1.wav", "demo_2.wav"])
    def test_demo_audio_hybrid_parity_with_ai(self, pipeline: AudioProcessingPipeline, wav_filename: str):
        """Verify Hybrid engine preserves speech and matches AI-only performance on demo audio."""
        repo_root = Path(__file__).resolve().parent.parent
        wav_path = repo_root / wav_filename
        if not wav_path.exists():
            pytest.skip(f"{wav_filename} not found in repository root")

        with open(wav_path, "rb") as f:
            raw_bytes = f.read()

        audio_in, noise_ref, sr = pipeline.decode_audio_bytes(raw_bytes)
        assert noise_ref is None, f"{wav_filename} stereo channels should downmix to mono"

        res_ai = pipeline.process(
            audio_in, sample_rate=sr, mode="ai_only", model_name="dtln_quantized"
        )
        res_hyb = pipeline.process(
            audio_in, sample_rate=sr, mode="hybrid", model_name="dtln_quantized"
        )

        assert res_ai.success is True
        assert res_hyb.success is True

        # Ensure output is fully audible and not cancelled out
        assert res_hyb.metrics.output_rms > 0.02
        assert res_hyb.metrics.estimated_attenuation_db > 0.0

        # Parity check: hybrid output RMS should be within 10% of AI-only
        rms_ratio = res_hyb.metrics.output_rms / res_ai.metrics.output_rms
        assert 0.90 <= rms_ratio <= 1.10, (
            f"Hybrid RMS ({res_hyb.metrics.output_rms}) deviated from AI-only RMS ({res_ai.metrics.output_rms})"
        )

        # Attenuation parity: within 1.0 dB of AI-only
        atten_diff = abs(res_hyb.metrics.estimated_attenuation_db - res_ai.metrics.estimated_attenuation_db)
        assert atten_diff < 1.0, f"Attenuation diff {atten_diff:.2f} dB too high"

    def test_crosstalk_guard_in_pipeline(self, pipeline: AudioProcessingPipeline):
        """Passing an identical speech reference must not destroy primary speech."""
        sr = 16000
        t = np.linspace(0, 0.5, int(sr * 0.5))
        speech = 0.4 * np.sin(2 * np.pi * 300 * t) + 0.1 * np.random.randn(len(t))

        # Pass identical signal as noise_reference
        res = pipeline.process(
            audio_in=speech,
            sample_rate=sr,
            mode="hybrid",
            model_name="dtln_quantized",
            noise_reference=speech,
        )

        assert res.success is True
        # Output RMS must remain strong, not collapsed to zero
        assert res.metrics.output_rms > 0.02
        assert res.metrics.estimated_attenuation_db > 0.0

    def test_crosstalk_guard_in_streaming_hybrid_engine(self):
        """HybridEngine must bypass ANC when reference is heavily correlated with measured."""
        model = SpectralGateModel(sample_rate=16_000)
        anc_cfg = FrameANCConfig(filter_length=32, step_size=0.01)
        sec_path = np.zeros(32, dtype=np.float64)
        sec_path[0] = 1.0

        engine = HybridEngine(
            model=model,
            anc_config=anc_cfg,
            secondary_path_true=sec_path,
            secondary_path_model=sec_path,
            sample_rate=16_000,
        )

        # Identical frame (100% crosstalk)
        x = np.sin(np.linspace(0, 10, 320)).astype(np.float64) * 0.3
        enhanced, timing = engine.process_frame(measured=x, reference=x)

        assert len(enhanced) == len(x)
        # Enhanced signal must maintain energy, not be destroyed
        assert np.sqrt(np.mean(enhanced ** 2)) > 0.01
