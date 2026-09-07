"""Tests for the overlap-add streaming processor (Phase 7)."""

from __future__ import annotations

import numpy as np
import pytest

from ai.models.base import EnhancementModel
from ai.streaming.overlap_add import OverlapAddProcessor


class _IdentityModel(EnhancementModel):
    """Pass-through for testing OLA reconstruction."""

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
        return self.validate_input(x).copy()


class TestOverlapAddReconstruction:
    """Verify that OLA with an identity model reproduces the input."""

    def test_known_signal_reconstruction(self) -> None:
        """A sine wave through identity OLA should match the original."""
        model = _IdentityModel()
        ola = OverlapAddProcessor(model, frame_size=256, overlap=0.5)

        sr = 16_000
        t = np.arange(sr) / sr
        signal = np.sin(2 * np.pi * 440 * t)

        output = ola.process(signal)
        assert len(output) == len(signal)
        # Interior samples should match closely (edges may have OLA artifacts)
        margin = ola.hop_size
        np.testing.assert_allclose(
            output[margin:-margin], signal[margin:-margin], atol=0.05
        )

    def test_arbitrary_length(self) -> None:
        """Signal length not a multiple of frame size should still work."""
        model = _IdentityModel()
        ola = OverlapAddProcessor(model, frame_size=256, overlap=0.5)

        # Odd length that's not a multiple of anything
        signal = np.random.RandomState(42).randn(1337)
        output = ola.process(signal)
        assert len(output) == len(signal)

    def test_short_signal(self) -> None:
        """Signal shorter than frame size should be handled."""
        model = _IdentityModel()
        ola = OverlapAddProcessor(model, frame_size=512, overlap=0.5)

        signal = np.array([1.0, 2.0, 3.0])
        output = ola.process(signal)
        assert len(output) == len(signal)

    def test_empty_signal(self) -> None:
        """Empty input should return empty output."""
        model = _IdentityModel()
        ola = OverlapAddProcessor(model, frame_size=256, overlap=0.5)
        output = ola.process(np.array([], dtype=np.float64))
        assert len(output) == 0

    def test_silent_regions_preserved(self) -> None:
        """Silent portions of the signal should remain near-zero."""
        model = _IdentityModel()
        ola = OverlapAddProcessor(model, frame_size=256, overlap=0.5)

        signal = np.zeros(2000, dtype=np.float64)
        # Only middle portion has content
        signal[800:1200] = np.sin(2 * np.pi * 440 * np.arange(400) / 16000)

        output = ola.process(signal)
        assert len(output) == len(signal)
        # Silent portions should be near zero
        assert np.max(np.abs(output[:600])) < 0.05
        assert np.max(np.abs(output[1400:])) < 0.05


class TestOverlapAddStreaming:
    """Verify the stateful streaming interface."""

    def test_streaming_produces_output(self) -> None:
        model = _IdentityModel()
        ola = OverlapAddProcessor(model, frame_size=256, overlap=0.5)

        signal = np.random.RandomState(42).randn(2048)
        chunk_size = 128
        outputs = []
        for i in range(0, len(signal), chunk_size):
            chunk = signal[i:i + chunk_size]
            out = ola.process_chunk(chunk)
            if len(out) > 0:
                outputs.append(out)

        assert len(outputs) > 0, "Streaming should produce some output"
        total_output = np.concatenate(outputs)
        assert len(total_output) > 0

    def test_reset_clears_state(self) -> None:
        model = _IdentityModel()
        ola = OverlapAddProcessor(model, frame_size=256, overlap=0.5)

        ola.process_chunk(np.ones(512))
        ola.reset()
        assert len(ola._input_buffer) == 0


class TestOverlapAddValidation:
    """Input validation tests."""

    def test_rejects_non_model(self) -> None:
        with pytest.raises(TypeError):
            OverlapAddProcessor("not a model")  # type: ignore

    def test_rejects_2d_input(self) -> None:
        model = _IdentityModel()
        ola = OverlapAddProcessor(model, frame_size=256)
        with pytest.raises(ValueError):
            ola.process(np.zeros((10, 2)))
