from __future__ import annotations

import numpy as np
import pytest

from anc.calibration import (
    align_signals,
    apply_known_delay,
    estimate_delay,
)


def make_test_signal(
    length: int = 2_000,
) -> np.ndarray:
    """Create a deterministic non-periodic calibration signal."""

    rng = np.random.default_rng(
        26052
    )

    white = rng.normal(
        0.0,
        1.0,
        size=length,
    )

    output = np.empty(
        length,
        dtype=np.float64,
    )

    previous = 0.0

    for index, value in enumerate(
        white
    ):

        previous = (
            0.82
            * previous
            + 0.18
            * value
        )

        output[index] = (
            previous
        )

    return output


def test_estimate_positive_delay() -> None:
    reference = make_test_signal()

    expected_delay = 73

    target = apply_known_delay(
        reference,
        delay_samples=(
            expected_delay
        ),
    )

    estimate = estimate_delay(
        reference,
        target,
        max_delay_samples=200,
        minimum_overlap_samples=500,
    )

    assert (
        estimate.lag_samples
        == expected_delay
    )


def test_estimate_negative_delay() -> None:
    reference = make_test_signal()

    expected_delay = -41

    target = apply_known_delay(
        reference,
        delay_samples=(
            expected_delay
        ),
    )

    estimate = estimate_delay(
        reference,
        target,
        max_delay_samples=200,
        minimum_overlap_samples=500,
    )

    assert (
        estimate.lag_samples
        == expected_delay
    )


def test_alignment_removes_known_delay() -> None:
    reference = make_test_signal()

    expected_delay = 64

    target = apply_known_delay(
        reference,
        delay_samples=(
            expected_delay
        ),
    )

    result = align_signals(
        reference,
        target,
        max_delay_samples=200,
        minimum_overlap_samples=500,
    )

    assert (
        result.delay.lag_samples
        == expected_delay
    )

    np.testing.assert_allclose(
        result.reference,
        result.target,
        atol=1e-12,
    )


def test_alignment_handles_polarity_inversion() -> None:
    reference = make_test_signal()

    expected_delay = 35

    target = (
        -apply_known_delay(
            reference,
            delay_samples=(
                expected_delay
            ),
        )
    )

    result = align_signals(
        reference,
        target,
        max_delay_samples=200,
        minimum_overlap_samples=500,
        allow_polarity_inversion=True,
    )

    assert (
        result.delay.lag_samples
        == expected_delay
    )

    assert (
        result.delay.normalized_peak_correlation
        < 0.0
    )


def test_maximum_delay_limit_is_respected() -> None:
    reference = make_test_signal()

    target = apply_known_delay(
        reference,
        delay_samples=120,
    )

    estimate = estimate_delay(
        reference,
        target,
        max_delay_samples=50,
        minimum_overlap_samples=500,
    )

    assert (
        abs(
            estimate.lag_samples
        )
        <= 50
    )


def test_nonfinite_signal_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        estimate_delay(
            np.array(
                [1.0, np.nan]
            ),
            np.array(
                [1.0, 2.0]
            ),
        )


def test_alignment_requires_sufficient_overlap() -> None:
    with pytest.raises(
        ValueError,
        match="minimum overlap",
    ):
        align_signals(
            np.arange(
                10,
                dtype=np.float64,
            ),
            np.arange(
                10,
                dtype=np.float64,
            ),
            max_delay_samples=5,
            minimum_overlap_samples=20,
        )