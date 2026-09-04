import numpy as np
import pytest

from anc.statistics import (
    align_impulse_responses,
    coefficient_correlation,
    coefficient_error,
    coefficient_mse,
    coefficient_rmse,
    relative_coefficient_error,
)


def test_align_equal_length_responses() -> None:
    true_h = np.array(
        [1.0, 2.0, 3.0]
    )

    estimated_h = np.array(
        [4.0, 5.0, 6.0]
    )

    aligned_true, aligned_estimated = (
        align_impulse_responses(
            true_h,
            estimated_h,
        )
    )

    assert np.allclose(
        aligned_true,
        true_h,
    )

    assert np.allclose(
        aligned_estimated,
        estimated_h,
    )


def test_align_shorter_true_response() -> None:
    true_h = np.array(
        [1.0, 2.0]
    )

    estimated_h = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    aligned_true, aligned_estimated = (
        align_impulse_responses(
            true_h,
            estimated_h,
        )
    )

    expected_true = np.array(
        [1.0, 2.0, 0.0, 0.0]
    )

    assert np.allclose(
        aligned_true,
        expected_true,
    )

    assert np.allclose(
        aligned_estimated,
        estimated_h,
    )


def test_align_shorter_estimated_response() -> None:
    true_h = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    estimated_h = np.array(
        [1.0, 2.0]
    )

    aligned_true, aligned_estimated = (
        align_impulse_responses(
            true_h,
            estimated_h,
        )
    )

    expected_estimated = np.array(
        [1.0, 2.0, 0.0, 0.0]
    )

    assert np.allclose(
        aligned_true,
        true_h,
    )

    assert np.allclose(
        aligned_estimated,
        expected_estimated,
    )


def test_coefficient_error() -> None:
    true_h = np.array(
        [1.0, 2.0, 3.0]
    )

    estimated_h = np.array(
        [0.5, 2.5, 2.0]
    )

    expected = np.array(
        [0.5, -0.5, 1.0]
    )

    assert np.allclose(
        coefficient_error(
            true_h,
            estimated_h,
        ),
        expected,
    )


def test_coefficient_mse_identical_filters() -> None:
    h = np.array(
        [1.0, 2.0, 3.0]
    )

    assert coefficient_mse(
        h,
        h,
    ) == pytest.approx(
        0.0
    )


def test_coefficient_mse_known_value() -> None:
    true_h = np.array(
        [1.0, 2.0]
    )

    estimated_h = np.array(
        [2.0, 4.0]
    )

    expected = (
        1.0 + 4.0
    ) / 2.0

    assert coefficient_mse(
        true_h,
        estimated_h,
    ) == pytest.approx(
        expected
    )


def test_rmse_is_square_root_of_mse() -> None:
    true_h = np.array(
        [1.0, 2.0, 3.0]
    )

    estimated_h = np.array(
        [2.0, 2.0, 1.0]
    )

    mse = coefficient_mse(
        true_h,
        estimated_h,
    )

    rmse = coefficient_rmse(
        true_h,
        estimated_h,
    )

    assert rmse == pytest.approx(
        np.sqrt(mse)
    )


def test_relative_error_identical_filters() -> None:
    h = np.array(
        [1.0, 2.0, 3.0]
    )

    assert relative_coefficient_error(
        h,
        h,
    ) == pytest.approx(
        0.0
    )


def test_relative_error_known_value() -> None:
    true_h = np.array(
        [3.0, 4.0]
    )

    estimated_h = np.array(
        [0.0, 0.0]
    )

    assert relative_coefficient_error(
        true_h,
        estimated_h,
    ) == pytest.approx(
        1.0
    )


def test_identical_nonconstant_filters_have_correlation_one() -> None:
    h = np.array(
        [1.0, 2.0, 4.0, 8.0]
    )

    assert coefficient_correlation(
        h,
        h,
    ) == pytest.approx(
        1.0
    )


def test_rejects_empty_impulse_response() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        align_impulse_responses(
            np.array([]),
            np.array([1.0]),
        )