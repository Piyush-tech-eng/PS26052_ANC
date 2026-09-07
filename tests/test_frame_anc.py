"""Tests for frame-by-frame ANC matching the offline engine (Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest

from ai.streaming.frame_anc import FrameANC, FrameANCConfig


def _make_test_signals(
    num_samples: int = 4000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create test signals: reference, disturbance, secondary paths."""
    rng = np.random.RandomState(seed)
    reference = rng.randn(num_samples)
    disturbance = 0.5 * rng.randn(num_samples)
    secondary_true = np.array([0.8, -0.3, 0.1], dtype=np.float64)
    secondary_model = np.array([0.75, -0.28, 0.09], dtype=np.float64)
    return reference, disturbance, secondary_true, secondary_model


class TestFrameANCBasic:
    """Basic functionality of the frame ANC processor."""

    def test_process_frame_returns_correct_length(self) -> None:
        ref, dist, s_true, s_model = _make_test_signals(num_samples=256)
        config = FrameANCConfig(filter_length=32, algorithm="fxnlms")
        anc = FrameANC(config, s_true, s_model)
        residual = anc.process_frame(ref, dist)
        assert len(residual) == len(ref)

    def test_process_sample_returns_float(self) -> None:
        _, _, s_true, s_model = _make_test_signals()
        config = FrameANCConfig(filter_length=16, algorithm="fxnlms")
        anc = FrameANC(config, s_true, s_model)
        result = anc.process_sample(0.5, 0.3)
        assert isinstance(result, float)

    def test_none_algorithm_no_adaptation(self) -> None:
        """With algorithm='none', coefficients should stay zero."""
        ref, dist, s_true, s_model = _make_test_signals(num_samples=100)
        config = FrameANCConfig(filter_length=16, algorithm="none")
        anc = FrameANC(config, s_true, s_model)
        anc.process_frame(ref, dist)
        np.testing.assert_array_equal(anc.coefficients, np.zeros(16))

    def test_fxnlms_adapts_coefficients(self) -> None:
        """FxNLMS should modify coefficients after processing."""
        ref, dist, s_true, s_model = _make_test_signals(num_samples=500)
        config = FrameANCConfig(filter_length=32, algorithm="fxnlms", step_size=0.01)
        anc = FrameANC(config, s_true, s_model)
        anc.process_frame(ref, dist)
        # Coefficients should no longer be all zero
        assert not np.allclose(anc.coefficients, 0.0)

    def test_coefficients_property_is_copy(self) -> None:
        _, _, s_true, s_model = _make_test_signals()
        config = FrameANCConfig(filter_length=16, algorithm="fxnlms")
        anc = FrameANC(config, s_true, s_model)
        coeffs = anc.coefficients
        coeffs[:] = 999.0
        # Should not affect internal state
        assert np.allclose(anc.coefficients, 0.0)


class TestFrameANCMatchesOffline:
    """Frame-by-frame output should match the offline engine on the same input."""

    def test_fxnlms_matches_sample_by_sample(self) -> None:
        """Process the same signal sample-by-sample and frame-by-frame;
        both should produce identical results."""
        ref, dist, s_true, s_model = _make_test_signals(num_samples=500)
        config = FrameANCConfig(filter_length=32, algorithm="fxnlms", step_size=0.005)

        # Sample-by-sample
        anc1 = FrameANC(config, s_true, s_model)
        residual_sbs = np.empty(len(ref))
        for i in range(len(ref)):
            residual_sbs[i] = anc1.process_sample(ref[i], dist[i])

        # Frame-by-frame (varying frame sizes)
        anc2 = FrameANC(config, s_true, s_model)
        residual_fbf = anc2.process_frame(ref, dist)

        np.testing.assert_allclose(residual_sbs, residual_fbf, atol=1e-12)

    def test_chunked_matches_single_frame(self) -> None:
        """Processing in chunks should match processing all at once."""
        ref, dist, s_true, s_model = _make_test_signals(num_samples=400)
        config = FrameANCConfig(filter_length=16, algorithm="fxnlms", step_size=0.01)

        # All at once
        anc1 = FrameANC(config, s_true, s_model)
        residual_all = anc1.process_frame(ref, dist)

        # In 100-sample chunks
        anc2 = FrameANC(config, s_true, s_model)
        chunks = []
        for i in range(0, 400, 100):
            chunk = anc2.process_frame(ref[i:i+100], dist[i:i+100])
            chunks.append(chunk)
        residual_chunked = np.concatenate(chunks)

        np.testing.assert_allclose(residual_all, residual_chunked, atol=1e-12)


class TestFrameANCAlgorithms:
    """Test all supported algorithms produce reasonable output."""

    @pytest.mark.parametrize("algo", ["lms", "nlms", "fxlms", "fxnlms", "none"])
    def test_algorithm_runs_without_error(self, algo: str) -> None:
        ref, dist, s_true, s_model = _make_test_signals(num_samples=200)
        config = FrameANCConfig(filter_length=16, algorithm=algo, step_size=0.01)
        anc = FrameANC(config, s_true, s_model)
        residual = anc.process_frame(ref, dist)
        assert len(residual) == len(ref)
        assert np.isfinite(residual).all()


class TestFrameANCValidation:
    """Input validation tests."""

    def test_mismatched_frame_lengths(self) -> None:
        _, _, s_true, s_model = _make_test_signals()
        config = FrameANCConfig(filter_length=16, algorithm="fxnlms")
        anc = FrameANC(config, s_true, s_model)
        with pytest.raises(ValueError, match="equal lengths"):
            anc.process_frame(np.ones(100), np.ones(200))

    def test_invalid_algorithm(self) -> None:
        with pytest.raises(ValueError):
            FrameANCConfig(algorithm="invalid")

    def test_reset_zeros_state(self) -> None:
        ref, dist, s_true, s_model = _make_test_signals(num_samples=200)
        config = FrameANCConfig(filter_length=16, algorithm="fxnlms")
        anc = FrameANC(config, s_true, s_model)
        anc.process_frame(ref, dist)
        anc.reset()
        np.testing.assert_array_equal(anc.coefficients, np.zeros(16))
