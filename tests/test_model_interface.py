"""Tests for the EnhancementModel interface contract (Phase 5).

Any model behind ``base.py`` must satisfy the ``enhance(x) -> y`` shape/dtype
contract and expose the required properties.
"""

from __future__ import annotations

import numpy as np
import pytest

from ai.models.base import EnhancementModel
from ai.models.spectral_gate import SpectralGateModel


# ---------------------------------------------------------------------------
# A minimal concrete subclass for testing the ABC itself
# ---------------------------------------------------------------------------


class _IdentityModel(EnhancementModel):
    """Pass-through model for contract testing."""

    @property
    def name(self) -> str:
        return "identity"

    @property
    def sample_rate(self) -> int:
        return 16_000

    @property
    def frame_size(self) -> int:
        return 0

    def enhance(self, x: np.ndarray) -> np.ndarray:
        x = self.validate_input(x)
        return x.copy()


# ---------------------------------------------------------------------------
# Interface contract tests (apply to any EnhancementModel)
# ---------------------------------------------------------------------------


def _check_contract(model: EnhancementModel) -> None:
    """Assert the basic contract for an EnhancementModel instance."""
    # Properties
    assert isinstance(model.name, str) and len(model.name) > 0
    assert isinstance(model.sample_rate, int) and model.sample_rate > 0
    assert isinstance(model.frame_size, int) and model.frame_size >= 0

    # enhance() with a normal signal
    x = np.random.RandomState(42).randn(8000).astype(np.float64)
    y = model.enhance(x)
    assert isinstance(y, np.ndarray)
    assert y.ndim == 1
    assert len(y) == len(x), f"Output length {len(y)} != input length {len(x)}"
    assert y.dtype == np.float64
    assert np.isfinite(y).all(), "Output contains NaN or Inf"

    # enhance() with a short signal
    x_short = np.array([0.1, -0.2, 0.3], dtype=np.float64)
    y_short = model.enhance(x_short)
    assert len(y_short) == len(x_short)

    # enhance() rejects 2-D input
    with pytest.raises(ValueError, match="one-dimensional"):
        model.enhance(np.zeros((10, 2)))

    # enhance() rejects empty input
    with pytest.raises(ValueError, match="empty"):
        model.enhance(np.array([], dtype=np.float64))


class TestIdentityModel:
    """Verify the ABC contract via the identity pass-through."""

    def test_contract(self) -> None:
        _check_contract(_IdentityModel())

    def test_identity_output(self) -> None:
        model = _IdentityModel()
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(model.enhance(x), x)

    def test_repr(self) -> None:
        model = _IdentityModel()
        r = repr(model)
        assert "identity" in r
        assert "16000" in r


class TestSpectralGateContract:
    """Verify the spectral gate model satisfies the full interface contract."""

    def test_contract(self) -> None:
        _check_contract(SpectralGateModel(sample_rate=8000))

    def test_contract_16k(self) -> None:
        _check_contract(SpectralGateModel(sample_rate=16_000))


class TestSpectralGateDenoising:
    """Verify the spectral gate actually reduces noise."""

    def test_improves_snr_on_noisy_signal(self) -> None:
        """A noisy signal should come out cleaner than it went in."""
        from anc.evaluation.metrics import compute_si_snr

        rng = np.random.RandomState(42)
        sr = 16_000

        # Create a speech-like AM signal with additive white noise
        t = np.arange(sr) / sr  # 1 second
        envelope = np.maximum(0, np.sin(2 * np.pi * 4 * t))  # syllable-rate AM
        clean = 0.5 * envelope * np.sin(2 * np.pi * 440 * t)
        noise = 0.3 * rng.randn(sr)
        noisy = clean + noise

        model = SpectralGateModel(sample_rate=sr, threshold_db=3.0)
        enhanced = model.enhance(noisy)

        # SI-SNR should improve (higher is better)
        si_snr_before = compute_si_snr(noisy, clean)
        si_snr_after = compute_si_snr(enhanced, clean)
        assert si_snr_after > si_snr_before, (
            f"SI-SNR should improve: before={si_snr_before:.1f} dB, "
            f"after={si_snr_after:.1f} dB"
        )

    def test_preserves_clean_signal(self) -> None:
        """A clean signal should not be significantly degraded."""
        sr = 16_000
        t = np.arange(sr) / sr
        clean = 0.5 * np.sin(2 * np.pi * 440 * t)

        model = SpectralGateModel(sample_rate=sr)
        enhanced = model.enhance(clean)

        # Should not introduce significant distortion
        correlation = np.corrcoef(clean, enhanced)[0, 1]
        assert correlation > 0.85, f"Clean signal correlation dropped to {correlation}"

    def test_reset_clears_noise_floor(self) -> None:
        model = SpectralGateModel(sample_rate=8000)
        x = np.random.randn(4000)
        model.enhance(x)
        assert model._noise_floor is not None
        model.reset()
        assert model._noise_floor is None

    def test_invalid_params_rejected(self) -> None:
        with pytest.raises(ValueError):
            SpectralGateModel(sample_rate=-1)
        with pytest.raises(ValueError):
            SpectralGateModel(fft_size=0)
        with pytest.raises(ValueError):
            SpectralGateModel(noise_percentile=0)
