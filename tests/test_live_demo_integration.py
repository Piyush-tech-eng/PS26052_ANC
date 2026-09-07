"""Integration tests for the live demo pipeline components.

Tests the full wiring: model factory → HybridEngine → audio processing,
and verifies the latency monitor reports plausible numbers.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from ai.models import get_best_available_model
from ai.models.base import EnhancementModel
from ai.models.spectral_gate import SpectralGateModel
from ai.streaming.hybrid_engine import HybridEngine
from ai.hardware.latency_monitor import LatencyMonitor


class TestModelFactory:
    """Tests for get_best_available_model()."""

    def test_auto_returns_enhancement_model(self):
        model = get_best_available_model()
        assert isinstance(model, EnhancementModel)
        assert model.name in ("dtln", "rnnoise", "spectral_gate")

    def test_prefer_spectral_gate(self):
        model = get_best_available_model(prefer="spectral_gate")
        assert model.name == "spectral_gate"

    def test_prefer_dtln_if_available(self):
        """DTLN should be available since we installed onnxruntime + checkpoint."""
        from ai.models.pretrained_dtln import is_available
        if not is_available():
            pytest.skip("onnxruntime not installed")
        model = get_best_available_model(prefer="dtln")
        assert model.name == "dtln"

    def test_auto_prefers_dtln_over_spectral_gate(self):
        """When DTLN is available, auto should pick it."""
        from ai.models.pretrained_dtln import is_available, _find_model_path
        if not is_available() or _find_model_path() is None:
            pytest.skip("DTLN not fully available")
        model = get_best_available_model()
        assert model.name == "dtln"


class TestHybridEngineIntegration:
    """Tests that the hybrid engine works with real models."""

    @pytest.fixture
    def engine_spectral_gate(self):
        model = SpectralGateModel(sample_rate=16_000)
        return HybridEngine(model=model, sample_rate=16_000)

    @pytest.fixture
    def engine_dtln(self):
        from ai.models.pretrained_dtln import is_available, _find_model_path
        if not is_available() or _find_model_path() is None:
            pytest.skip("DTLN not available")
        model = get_best_available_model(prefer="dtln")
        return HybridEngine(model=model, sample_rate=16_000)

    def test_spectral_gate_engine_processes(self, engine_spectral_gate):
        x = np.random.default_rng(42).standard_normal(3200).astype(np.float64) * 0.3
        enhanced, timing = engine_spectral_gate.process_frame(x)
        assert len(enhanced) == len(x)
        assert timing.total_seconds > 0
        assert timing.realtime_ratio > 0

    def test_dtln_engine_processes(self, engine_dtln):
        x = np.random.default_rng(42).standard_normal(3200).astype(np.float64) * 0.3
        enhanced, timing = engine_dtln.process_frame(x)
        assert len(enhanced) == len(x)
        assert timing.total_seconds > 0

    def test_dtln_suppresses_noise(self, engine_dtln):
        """DTLN should suppress pure noise (no speech content)."""
        rng = np.random.default_rng(42)
        noise = rng.standard_normal(16000).astype(np.float64) * 0.5
        enhanced, _ = engine_dtln.process_batch(noise)
        # RMS should decrease significantly for pure noise
        rms_in = np.sqrt(np.mean(noise**2))
        rms_out = np.sqrt(np.mean(enhanced**2))
        assert rms_out < rms_in * 0.5, (
            f"DTLN should suppress noise: in={rms_in:.4f}, out={rms_out:.4f}"
        )

    def test_engine_reset(self, engine_spectral_gate):
        x = np.random.default_rng(42).standard_normal(1600).astype(np.float64)
        engine_spectral_gate.process_frame(x)
        assert engine_spectral_gate.latest_timing is not None
        engine_spectral_gate.reset()
        # After reset, no timing history
        assert engine_spectral_gate.latest_timing is None


class TestLatencyMonitorIntegration:
    """Tests that latency monitor works in a realistic pipeline context."""

    def test_monitor_tracks_stages(self):
        monitor = LatencyMonitor()
        monitor.begin()
        time.sleep(0.001)
        monitor.mark("receive")
        time.sleep(0.001)
        monitor.mark("enhance")
        time.sleep(0.001)
        monitor.mark("playback")
        snapshot = monitor.end()

        assert "receive" in snapshot.stages
        assert "enhance" in snapshot.stages
        assert "playback" in snapshot.stages
        assert snapshot.total_ms > 0

    def test_monitor_rolling_average(self):
        monitor = LatencyMonitor()
        for _ in range(5):
            monitor.begin()
            time.sleep(0.001)
            monitor.mark("process")
            monitor.end()

        avg = monitor.average_ms
        assert "process" in avg
        assert avg["process"] > 0
        assert monitor.average_total_ms > 0


class TestStandardMetrics:
    """Tests that standard metrics (pystoi) integrate correctly."""

    def test_stoi_standard_uses_pystoi(self):
        from anc.evaluation.metrics import compute_stoi_standard
        rng = np.random.default_rng(42)
        ref = rng.standard_normal(16000).astype(np.float64) * 0.3
        est = ref + rng.standard_normal(16000) * 0.1
        value, source = compute_stoi_standard(est, ref, 16000)
        # pystoi is installed, so source should be "pystoi"
        try:
            import pystoi
            assert source == "pystoi"
        except ImportError:
            assert source == "approx"
        assert -1.0 <= value <= 1.0

    def test_pesq_standard_falls_back(self):
        from anc.evaluation.metrics import compute_pesq_standard
        rng = np.random.default_rng(42)
        ref = rng.standard_normal(16000).astype(np.float64) * 0.3
        est = ref + rng.standard_normal(16000) * 0.1
        value, source = compute_pesq_standard(est, ref, 16000)
        # pesq may not be installed (needs MSVC), so accept either source
        assert source in ("pesq", "approx")
        assert 0.5 <= value <= 5.0
