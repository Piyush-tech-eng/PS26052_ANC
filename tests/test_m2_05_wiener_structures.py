import numpy as np
import pytest

from anc.statistics.wiener import (
    build_correlation_matrix,
    build_cross_correlation_vector,
)


def test_build_correlation_matrix() -> None:
    rxx = np.array(
        [10.0, 4.0, 2.0, 1.0]
    )

    matrix = build_correlation_matrix(
        rxx,
        filter_length=4,
    )

    expected = np.array(
        [
            [10.0, 4.0, 2.0, 1.0],
            [4.0, 10.0, 4.0, 2.0],
            [2.0, 4.0, 10.0, 4.0],
            [1.0, 2.0, 4.0, 10.0],
        ]
    )

    assert np.allclose(
        matrix,
        expected,
    )


def test_correlation_matrix_is_symmetric() -> None:
    rxx = np.array(
        [5.0, 2.0, 1.0]
    )

    matrix = build_correlation_matrix(
        rxx,
        filter_length=3,
    )

    assert np.allclose(
        matrix,
        matrix.T,
    )


def test_correlation_matrix_has_correct_shape() -> None:
    rxx = np.arange(
        10,
        dtype=np.float64,
    )

    matrix = build_correlation_matrix(
        rxx,
        filter_length=6,
    )

    assert matrix.shape == (6, 6)


def test_correlation_matrix_requires_enough_lags() -> None:
    rxx = np.array(
        [1.0, 0.5]
    )

    with pytest.raises(
        ValueError,
        match="Not enough autocorrelation",
    ):
        build_correlation_matrix(
            rxx,
            filter_length=3,
        )


def test_cross_correlation_vector() -> None:
    rxd = np.array(
        [10.0, 8.0, 4.0, 2.0]
    )

    vector = build_cross_correlation_vector(
        rxd,
        filter_length=3,
    )

    expected = np.array(
        [10.0, 8.0, 4.0]
    )

    assert np.allclose(
        vector,
        expected,
    )


def test_cross_correlation_vector_has_correct_shape() -> None:
    rxd = np.arange(
        10,
        dtype=np.float64,
    )

    vector = build_cross_correlation_vector(
        rxd,
        filter_length=5,
    )

    assert vector.shape == (5,)


def test_cross_correlation_vector_requires_enough_lags() -> None:
    rxd = np.array(
        [1.0, 0.5]
    )

    with pytest.raises(
        ValueError,
        match="Not enough cross-correlation",
    ):
        build_cross_correlation_vector(
            rxd,
            filter_length=3,
        )


def test_correlation_matrix_rejects_nonfinite_values() -> None:
    rxx = np.array(
        [1.0, np.nan, 0.5]
    )

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        build_correlation_matrix(
            rxx,
            filter_length=3,
        )


def test_cross_correlation_vector_rejects_nonfinite_values() -> None:
    rxd = np.array(
        [1.0, np.inf, 0.5]
    )

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        build_cross_correlation_vector(
            rxd,
            filter_length=3,
        )


def test_filter_length_must_be_positive() -> None:
    rxx = np.array(
        [1.0, 0.5, 0.25]
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        build_correlation_matrix(
            rxx,
            filter_length=0,
        )


