"""Tests for speech-perceptual metrics: SI-SNR, STOI, PESQ-approx (Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from anc.evaluation.metrics import (
    compute_si_snr,
    compute_stoi,
    compute_pesq_approx,
    evaluate_scenario,
)


class TestSiSnr:
    """Scale-Invariant Signal-to-Noise Ratio."""

    def test_perfect_reconstruction(self) -> None:
        """SI-SNR of identical signals should be very high."""
        target = np.random.RandomState(42).randn(8000)
        si_snr = compute_si_snr(target, target)
        assert si_snr > 100  # Effectively infinite

    def test_scaled_copy_has_high_si_snr(self) -> None:
        """SI-SNR is scale-invariant, so a scaled copy should still be high."""
        target = np.random.RandomState(42).randn(8000)
        estimate = 2.0 * target  # Scaled
        si_snr = compute_si_snr(estimate, target)
        assert si_snr > 100

    def test_noise_lowers_si_snr(self) -> None:
        """Adding noise should reduce SI-SNR."""
        rng = np.random.RandomState(42)
        target = rng.randn(8000)
        noise = rng.randn(8000) * 0.5
        estimate = target + noise
        si_snr = compute_si_snr(estimate, target)
        assert si_snr < 20  # Noisy
        assert si_snr > 0   # But still some signal

    def test_uncorrelated_signals_near_zero(self) -> None:
        """Independent signals should have SI-SNR near 0 or negative."""
        rng = np.random.RandomState(42)
        target = rng.randn(8000)
        estimate = rng.randn(8000)
        si_snr = compute_si_snr(estimate, target)
        assert si_snr < 5  # Very low

    def test_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="equal lengths"):
            compute_si_snr(np.ones(100), np.ones(200))


class TestStoi:
    """Short-Time Objective Intelligibility (approximation)."""

    def test_perfect_reconstruction(self) -> None:
        """Identical signals should yield STOI close to 1.0."""
        target = np.random.RandomState(42).randn(8000)
        stoi = compute_stoi(target, target, 8000)
        assert stoi > 0.95

    def test_noise_reduces_stoi(self) -> None:
        """Heavy noise should reduce STOI."""
        rng = np.random.RandomState(42)
        target = rng.randn(8000)
        estimate = target + 2.0 * rng.randn(8000)  # Heavy noise
        stoi = compute_stoi(estimate, target, 8000)
        assert stoi < 0.8

    def test_range_is_valid(self) -> None:
        """STOI should be in [-1, 1]."""
        rng = np.random.RandomState(42)
        target = rng.randn(8000)
        estimate = target + 0.1 * rng.randn(8000)
        stoi = compute_stoi(estimate, target, 8000)
        assert -1.0 <= stoi <= 1.0

    def test_short_signal(self) -> None:
        """Very short signals should return 0."""
        stoi = compute_stoi(np.ones(10), np.ones(10), 8000)
        assert stoi == 0.0


class TestPesqApprox:
    """Simplified PESQ-like quality estimate."""

    def test_perfect_reconstruction_high_quality(self) -> None:
        """Identical signals should yield high PESQ score."""
        target = np.random.RandomState(42).randn(8000)
        pesq = compute_pesq_approx(target, target, 8000)
        assert pesq > 3.0

    def test_noisy_signal_lower_quality(self) -> None:
        """Noisy estimate should have lower PESQ than clean."""
        rng = np.random.RandomState(42)
        target = rng.randn(8000)
        clean_pesq = compute_pesq_approx(target, target, 8000)
        noisy_estimate = target + rng.randn(8000)
        noisy_pesq = compute_pesq_approx(noisy_estimate, target, 8000)
        assert noisy_pesq < clean_pesq

    def test_range_is_valid(self) -> None:
        """PESQ-approx should be in [1.0, 4.5]."""
        rng = np.random.RandomState(42)
        target = rng.randn(8000)
        estimate = target + 0.5 * rng.randn(8000)
        pesq = compute_pesq_approx(estimate, target, 8000)
        assert 1.0 <= pesq <= 4.5


class TestEvaluateScenarioSpeechMetrics:
    """Verify evaluate_scenario includes speech metrics when target is present."""

    def test_no_target_returns_none_metrics(self) -> None:
        residual = np.random.randn(8000)
        metrics = evaluate_scenario(residual, sampling_rate_hz=8000)
        assert metrics["target_available"] is False
        assert metrics["si_snr_db"] is None
        assert metrics["stoi"] is None
        assert metrics["pesq_approx"] is None

    def test_with_target_returns_speech_metrics(self) -> None:
        rng = np.random.RandomState(42)
        target = rng.randn(8000)
        residual = target + 0.1 * rng.randn(8000)
        metrics = evaluate_scenario(residual, sampling_rate_hz=8000, target=target)
        assert metrics["target_available"] is True
        assert isinstance(metrics["si_snr_db"], float)
        assert isinstance(metrics["stoi"], float)
        assert isinstance(metrics["pesq_approx"], float)
        # Classical metrics should also be present
        assert isinstance(metrics["mse"], float)
        assert isinstance(metrics["rmse"], float)
