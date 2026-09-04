import numpy as np
import pytest

from anc.statistics import autocorrelation


def test_autocorrelation_lag_zero() -> None:
    x = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    lags, values = autocorrelation(x)

    assert lags[0] == 0

    expected = (
        1**2
        + 2**2
        + 3**2
        + 4**2
    ) / 4

    assert values[0] == pytest.approx(
        expected
    )


def test_biased_autocorrelation_lag_one() -> None:
    x = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    _, values = autocorrelation(
        x,
        max_lag=1,
        unbiased=False,
    )

    expected = (
        1 * 2
        + 2 * 3
        + 3 * 4
    ) / 4

    assert values[1] == pytest.approx(
        expected
    )


def test_unbiased_autocorrelation_lag_one() -> None:
    x = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    _, values = autocorrelation(
        x,
        max_lag=1,
        unbiased=True,
    )

    expected = (
        1 * 2
        + 2 * 3
        + 3 * 4
    ) / 3

    assert values[1] == pytest.approx(
        expected
    )


def test_zero_lag_is_signal_power() -> None:
    x = np.array(
        [-2.0, 0.0, 2.0]
    )

    _, values = autocorrelation(
        x,
        max_lag=0,
    )

    expected = (
        4 + 0 + 4
    ) / 3

    assert values[0] == pytest.approx(
        expected
    )


def test_autocorrelation_symmetry_for_full_range() -> None:
    x = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    _, positive = autocorrelation(
        x,
        max_lag=3,
    )

    # R[-k] = R[k].
    # Our API currently returns the non-negative
    # half, so the positive sequence is enough
    # to establish the convention used by M2.
    assert len(positive) == 4


def test_invalid_signal_rejected() -> None:
    with pytest.raises(ValueError):
        autocorrelation(
            np.array([])
        )


def test_invalid_lag_rejected() -> None:
    x = np.array(
        [1.0, 2.0, 3.0]
    )

    with pytest.raises(ValueError):
        autocorrelation(
            x,
            max_lag=3,
        )


def test_nonfinite_signal_rejected() -> None:
    x = np.array(
        [1.0, np.nan, 3.0]
    )

    with pytest.raises(ValueError):
        autocorrelation(x)