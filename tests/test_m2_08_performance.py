import numpy as np
import pytest

from anc.statistics import (
    explained_power_fraction,
    orthogonality_residual,
    relative_residual_power,
    signal_power,
)


def test_signal_power() -> None:
    x = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    expected = (
        1 + 4 + 9 + 16
    ) / 4

    assert signal_power(x) == pytest.approx(
        expected
    )


def test_orthogonality_at_zero_lag() -> None:
    error = np.array(
        [1.0, 2.0, 3.0]
    )

    reference = np.array(
        [2.0, 1.0, 0.0]
    )

    _, values = orthogonality_residual(
        error,
        reference,
        max_lag=0,
    )

    expected = (
        1 * 2
        + 2 * 1
        + 3 * 0
    ) / 3

    assert values[0] == pytest.approx(
        expected
    )


def test_orthogonality_zero_when_vectors_are_orthogonal() -> None:
    error = np.array(
        [1.0, 0.0, -1.0, 0.0]
    )

    reference = np.array(
        [0.0, 1.0, 0.0, 1.0]
    )

    _, values = orthogonality_residual(
        error,
        reference,
        max_lag=0,
    )

    assert values[0] == pytest.approx(
        0.0
    )


def test_relative_residual_power() -> None:
    target = np.array(
        [1.0, 1.0, 1.0, 1.0]
    )

    error = np.array(
        [0.5, 0.5, 0.5, 0.5]
    )

    # Target power = 1
    # Error power = 0.25
    assert relative_residual_power(
        target,
        error,
    ) == pytest.approx(
        0.25
    )


def test_explained_power_fraction() -> None:
    target = np.array(
        [1.0, 1.0, 1.0, 1.0]
    )

    error = np.array(
        [0.5, 0.5, 0.5, 0.5]
    )

    assert explained_power_fraction(
        target,
        error,
    ) == pytest.approx(
        0.75
    )


def test_orthogonality_requires_equal_lengths() -> None:
    with pytest.raises(
        ValueError,
        match="equal lengths",
    ):
        orthogonality_residual(
            np.ones(3),
            np.ones(4),
        )


def test_orthogonality_rejects_invalid_lag() -> None:
    with pytest.raises(
        ValueError,
        match="smaller than the number",
    ):
        orthogonality_residual(
            np.ones(3),
            np.ones(3),
            max_lag=3,
        )


def test_signal_power_rejects_empty_signal() -> None:
    with pytest.raises(ValueError):
        signal_power(
            np.array([])
        )