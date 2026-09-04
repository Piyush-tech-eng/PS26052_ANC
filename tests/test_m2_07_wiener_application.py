import numpy as np
import pytest

from anc.statistics import (
    apply_fir,
    mean_squared_error,
    normalized_correlation,
    root_mean_squared_error,
)


def test_apply_fir_matches_known_causal_result() -> None:
    x = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    h = np.array(
        [1.0, 0.5, -0.25]
    )

    output = apply_fir(
        x,
        h,
    )

    expected = np.array(
    [
        1.0,
        2.5,
        3.75,
        5.0,
        6.25,
    ]
)

    assert output == pytest.approx(
        expected
    )


def test_apply_fir_output_length_equals_input() -> None:
    x = np.arange(
        20,
        dtype=np.float64,
    )

    h = np.array(
        [0.2, 0.4, 0.1]
    )

    output = apply_fir(
        x,
        h,
    )

    assert output.shape == x.shape


def test_identity_filter() -> None:
    x = np.array(
        [2.0, -1.0, 5.0]
    )

    h = np.array(
        [1.0]
    )

    output = apply_fir(
        x,
        h,
    )

    assert output == pytest.approx(
        x
    )


def test_mse_zero_for_identical_signals() -> None:
    x = np.array(
        [1.0, 2.0, 3.0]
    )

    assert mean_squared_error(
        x,
        x,
    ) == pytest.approx(0.0)


def test_mse_known_value() -> None:
    target = np.array(
        [1.0, 2.0, 3.0]
    )

    estimate = np.array(
        [2.0, 2.0, 1.0]
    )

    expected = (
        1.0 + 0.0 + 4.0
    ) / 3.0

    assert mean_squared_error(
        target,
        estimate,
    ) == pytest.approx(
        expected
    )


def test_rmse_is_square_root_of_mse() -> None:
    target = np.array(
        [1.0, 2.0, 3.0]
    )

    estimate = np.array(
        [2.0, 2.0, 1.0]
    )

    mse = mean_squared_error(
        target,
        estimate,
    )

    rmse = root_mean_squared_error(
        target,
        estimate,
    )

    assert rmse == pytest.approx(
        np.sqrt(mse)
    )


def test_identical_nonconstant_signals_have_correlation_one() -> None:
    x = np.array(
        [1.0, 2.0, 4.0, 8.0]
    )

    assert normalized_correlation(
        x,
        x,
    ) == pytest.approx(
        1.0
    )


def test_reversed_signal_can_have_negative_correlation() -> None:
    x = np.array(
        [1.0, 2.0, 3.0]
    )

    y = -x

    assert normalized_correlation(
        x,
        y,
    ) == pytest.approx(
        -1.0
    )


def test_apply_fir_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValueError):
        apply_fir(
            np.ones((2, 2)),
            np.array([1.0]),
        )

    with pytest.raises(ValueError):
        apply_fir(
            np.ones(3),
            np.ones((2, 2)),
        )


def test_apply_fir_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        apply_fir(
            np.array([]),
            np.array([1.0]),
        )


def test_mse_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError):
        mean_squared_error(
            np.array([1.0, 2.0]),
            np.array([1.0]),
        )


def test_correlation_rejects_zero_variance() -> None:
    with pytest.raises(ValueError):
        normalized_correlation(
            np.ones(5),
            np.arange(5, dtype=np.float64),
        )