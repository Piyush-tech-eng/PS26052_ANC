import numpy as np
import pytest

from anc.adaptive import (
    analyze_lms_convergence,
    moving_average,
)


def test_moving_average_basic() -> None:

    values = np.array(
        [1.0, 2.0, 3.0, 4.0],
    )

    result = moving_average(
        values,
        window_size=2,
    )

    expected = np.array(
        [1.0, 1.5, 2.5, 3.5],
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_moving_average_window_larger_than_signal() -> None:

    values = np.array(
        [2.0, 4.0, 6.0],
    )

    result = moving_average(
        values,
        window_size=10,
    )

    expected = np.array(
        [2.0, 3.0, 4.0],
    )

    np.testing.assert_allclose(
        result,
        expected,
    )


def test_moving_average_invalid_window() -> None:

    with pytest.raises(
        ValueError
    ):

        moving_average(
            np.array(
                [1.0, 2.0],
            ),
            window_size=0,
        )


def test_convergence_detects_error_reduction() -> None:

    error = np.array(
        [4.0, 3.0, 2.0, 1.0],
    )

    coefficient_history = np.array(
        [
            [1.0],
            [1.5],
            [1.75],
            [1.80],
        ]
    )

    result = analyze_lms_convergence(
        error,
        coefficient_history,
        learning_window=2,
        stability_window=2,
    )

    assert result.error_decreased

    assert result.initial_error_power > (
        result.final_error_power
    )


def test_coefficient_change_is_correct() -> None:

    error = np.array(
        [1.0, 1.0, 1.0],
    )

    coefficient_history = np.array(
        [
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 7.0],
        ]
    )

    result = analyze_lms_convergence(
        error,
        coefficient_history,
    )

    expected = np.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 3.0],
        ]
    )

    np.testing.assert_allclose(
        result.coefficient_change,
        expected,
    )


def test_mismatched_history_length_rejected() -> None:

    with pytest.raises(
        ValueError
    ):

        analyze_lms_convergence(
            np.array(
                [1.0, 2.0],
            ),
            np.array(
                [
                    [1.0],
                    [2.0],
                    [3.0],
                ]
            ),
        )


def test_non_finite_error_rejected() -> None:

    with pytest.raises(
        ValueError
    ):

        analyze_lms_convergence(
            np.array(
                [1.0, np.nan],
            ),
            np.array(
                [
                    [1.0],
                    [2.0],
                ]
            ),
        )