import numpy as np
import pytest

from anc.adaptive import (
    WienerComparisonResult,
    compare_against_wiener,
)


def test_returns_expected_result_type() -> None:

    coefficient_history = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.5],
            [1.0, 1.0],
        ],
        dtype=np.float64,
    )

    wiener_coefficients = np.array(
        [1.0, 1.0],
        dtype=np.float64,
    )

    result = compare_against_wiener(
        coefficient_history,
        wiener_coefficients,
    )

    assert isinstance(
        result,
        WienerComparisonResult,
    )


def test_coefficient_error_is_correct() -> None:

    coefficient_history = np.array(
        [
            [0.0, 0.0],
            [1.0, 2.0],
            [2.0, 4.0],
        ],
        dtype=np.float64,
    )

    wiener_coefficients = np.array(
        [2.0, 3.0],
        dtype=np.float64,
    )

    result = compare_against_wiener(
        coefficient_history,
        wiener_coefficients,
    )

    expected = np.array(
        [
            [-2.0, -3.0],
            [-1.0, -1.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result.coefficient_error,
        expected,
    )


def test_coefficient_error_norm_is_correct() -> None:

    coefficient_history = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    wiener_coefficients = np.array(
        [0.0, 0.0],
        dtype=np.float64,
    )

    result = compare_against_wiener(
        coefficient_history,
        wiener_coefficients,
    )

    expected = np.array(
        [
            0.0,
            5.0,
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(
        result.coefficient_error_norm,
        expected,
    )


def test_detects_movement_toward_wiener_solution() -> None:

    coefficient_history = np.array(
        [
            [0.0],
            [2.0],
            [4.0],
            [5.0],
        ],
        dtype=np.float64,
    )

    wiener_coefficients = np.array(
        [5.0],
        dtype=np.float64,
    )

    result = compare_against_wiener(
        coefficient_history,
        wiener_coefficients,
        comparison_window=1,
    )

    assert result.moved_closer_to_wiener

    assert (
        result.final_coefficient_error
        < result.initial_coefficient_error
    )


def test_detects_movement_away_from_wiener_solution() -> None:

    coefficient_history = np.array(
        [
            [5.0],
            [4.0],
            [2.0],
            [0.0],
        ],
        dtype=np.float64,
    )

    wiener_coefficients = np.array(
        [5.0],
        dtype=np.float64,
    )

    result = compare_against_wiener(
        coefficient_history,
        wiener_coefficients,
        comparison_window=1,
    )

    assert not result.moved_closer_to_wiener

    assert (
        result.final_coefficient_error
        > result.initial_coefficient_error
    )


def test_minimum_error_and_index_are_correct() -> None:

    coefficient_history = np.array(
        [
            [0.0],
            [4.0],
            [5.0],
            [3.0],
        ],
        dtype=np.float64,
    )

    wiener_coefficients = np.array(
        [5.0],
        dtype=np.float64,
    )

    result = compare_against_wiener(
        coefficient_history,
        wiener_coefficients,
        comparison_window=1,
    )

    assert (
        result.minimum_coefficient_error
        == pytest.approx(
            0.0
        )
    )

    assert (
        result.minimum_error_index
        == 2
    )


def test_error_ratio_is_correct() -> None:

    coefficient_history = np.array(
        [
            [0.0],
            [5.0],
        ],
        dtype=np.float64,
    )

    wiener_coefficients = np.array(
        [5.0],
        dtype=np.float64,
    )

    result = compare_against_wiener(
        coefficient_history,
        wiener_coefficients,
        comparison_window=1,
    )

    assert (
        result.initial_coefficient_error
        == pytest.approx(
            5.0
        )
    )

    assert (
        result.final_coefficient_error
        == pytest.approx(
            0.0
        )
    )

    assert (
        result.coefficient_error_ratio
        == pytest.approx(
            0.0
        )
    )


def test_history_must_be_two_dimensional() -> None:

    with pytest.raises(
        ValueError
    ):

        compare_against_wiener(
            np.array(
                [1.0, 2.0],
            ),
            np.array(
                [1.0],
            ),
        )


def test_wiener_coefficients_must_be_one_dimensional() -> None:

    with pytest.raises(
        ValueError
    ):

        compare_against_wiener(
            np.array(
                [
                    [1.0],
                    [2.0],
                ]
            ),
            np.array(
                [
                    [1.0],
                ]
            ),
        )


def test_filter_lengths_must_match() -> None:

    with pytest.raises(
        ValueError
    ):

        compare_against_wiener(
            np.array(
                [
                    [1.0, 2.0],
                    [3.0, 4.0],
                ]
            ),
            np.array(
                [1.0],
            ),
        )


def test_empty_history_rejected() -> None:

    with pytest.raises(
        ValueError
    ):

        compare_against_wiener(
            np.empty(
                (
                    0,
                    2,
                )
            ),
            np.array(
                [1.0, 2.0],
            ),
        )


def test_empty_wiener_coefficients_rejected() -> None:

    with pytest.raises(
        ValueError
    ):

        compare_against_wiener(
            np.array(
                [
                    [1.0],
                ]
            ),
            np.array(
                [],
            ),
        )


def test_non_finite_history_rejected() -> None:

    with pytest.raises(
        ValueError
    ):

        compare_against_wiener(
            np.array(
                [
                    [1.0],
                    [np.nan],
                ]
            ),
            np.array(
                [1.0],
            ),
        )


def test_non_finite_wiener_coefficients_rejected() -> None:

    with pytest.raises(
        ValueError
    ):

        compare_against_wiener(
            np.array(
                [
                    [1.0],
                ]
            ),
            np.array(
                [np.inf],
            ),
        )


@pytest.mark.parametrize(
    "comparison_window",
    [
        0,
        -1,
    ],
)
def test_invalid_comparison_window_rejected(
    comparison_window: int,
) -> None:

    with pytest.raises(
        ValueError
    ):

        compare_against_wiener(
            np.array(
                [
                    [0.0],
                    [1.0],
                ]
            ),
            np.array(
                [1.0],
            ),
            comparison_window=(
                comparison_window
            ),
        )


def test_non_integer_comparison_window_rejected() -> None:

    with pytest.raises(
        TypeError
    ):

        compare_against_wiener(
            np.array(
                [
                    [0.0],
                    [1.0],
                ]
            ),
            np.array(
                [1.0],
            ),
            comparison_window=1.5,
        )