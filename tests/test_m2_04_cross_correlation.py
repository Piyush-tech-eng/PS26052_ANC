import numpy as np
import pytest

from anc.statistics import (
    cross_correlation,
)


def test_cross_correlation_lag_zero() -> None:
    """
    At lag zero:

        Rxd[0] = mean(d[n] x[n])
    """

    x = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    d = np.array(
        [2.0, 1.0, 0.0, -1.0]
    )

    lags, values = cross_correlation(
        x,
        d,
        max_lag=0,
    )

    expected = (
        1 * 2
        + 2 * 1
        + 3 * 0
        + 4 * -1
    ) / 4

    assert lags.tolist() == [0]

    assert values[0] == pytest.approx(
        expected
    )


def test_biased_cross_correlation_lag_one() -> None:
    """
    For lag 1:

        Rxd[1]
        =
        (1/N) *
        (
            d[1]x[0]
            +
            d[2]x[1]
            +
            d[3]x[2]
        )
    """

    x = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    d = np.array(
        [10.0, 20.0, 30.0, 40.0]
    )

    _, values = cross_correlation(
        x,
        d,
        max_lag=1,
        unbiased=False,
    )

    expected = (
        20 * 1
        + 30 * 2
        + 40 * 3
    ) / 4

    assert values[1] == pytest.approx(
        expected
    )


def test_unbiased_cross_correlation_lag_one() -> None:
    x = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    d = np.array(
        [10.0, 20.0, 30.0, 40.0]
    )

    _, values = cross_correlation(
        x,
        d,
        max_lag=1,
        unbiased=True,
    )

    expected = (
        20 * 1
        + 30 * 2
        + 40 * 3
    ) / 3

    assert values[1] == pytest.approx(
        expected
    )


def test_cross_correlation_with_identical_signals() -> None:
    """
    If d[n] = x[n], cross-correlation must reduce
    to autocorrelation under the same convention.
    """

    x = np.array(
        [1.0, -2.0, 3.0, 4.0]
    )

    _, values = cross_correlation(
        x,
        x,
        max_lag=3,
    )

    expected = np.array(
        [
            (
                1**2
                + (-2)**2
                + 3**2
                + 4**2
            ) / 4,

            (
                (-2) * 1
                + 3 * (-2)
                + 4 * 3
            ) / 4,

            (
                3 * 1
                + 4 * (-2)
            ) / 4,

            (
                4 * 1
            ) / 4,
        ],
        dtype=np.float64,
    )

    assert values == pytest.approx(
        expected
    )


def test_cross_correlation_rejects_unequal_lengths() -> None:
    x = np.array(
        [1.0, 2.0, 3.0]
    )

    d = np.array(
        [1.0, 2.0]
    )

    with pytest.raises(
        ValueError,
        match="equal lengths",
    ):
        cross_correlation(
            x,
            d,
        )


def test_cross_correlation_rejects_empty_signal() -> None:
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        cross_correlation(
            np.array([]),
            np.array([]),
        )


def test_cross_correlation_rejects_nonfinite_reference() -> None:
    x = np.array(
        [1.0, np.nan, 3.0]
    )

    d = np.array(
        [1.0, 2.0, 3.0]
    )

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        cross_correlation(
            x,
            d,
        )


def test_cross_correlation_rejects_nonfinite_desired() -> None:
    x = np.array(
        [1.0, 2.0, 3.0]
    )

    d = np.array(
        [1.0, np.inf, 3.0]
    )

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        cross_correlation(
            x,
            d,
        )


def test_cross_correlation_rejects_invalid_lag() -> None:
    x = np.array(
        [1.0, 2.0, 3.0]
    )

    d = np.array(
        [4.0, 5.0, 6.0]
    )

    with pytest.raises(
        ValueError,
        match="smaller than the number",
    ):
        cross_correlation(
            x,
            d,
            max_lag=3,
        )