import numpy as np
import pytest

from anc.statistics.basic import (
    average_power,
    mean,
    rms,
    summarize_signal_statistics,
    variance,
)


def test_mean() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])

    assert mean(x) == pytest.approx(2.5)


def test_population_variance() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])

    # Population variance:
    #
    # ((1-2.5)^2 + (2-2.5)^2
    #  + (3-2.5)^2 + (4-2.5)^2) / 4
    #
    # = 1.25

    assert variance(x) == pytest.approx(1.25)


def test_average_power() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0])

    expected = (1 + 4 + 9 + 16) / 4

    assert average_power(x) == pytest.approx(expected)


def test_rms_equals_square_root_of_power() -> None:
    x = np.array([-2.0, 0.0, 2.0, 4.0])

    power = average_power(x)
    rms_value = rms(x)

    assert rms_value == pytest.approx(
        np.sqrt(power)
    )


def test_power_equals_variance_plus_mean_squared() -> None:
    x = np.array([-1.0, 0.0, 2.0, 3.0])

    mu = mean(x)
    var = variance(x)
    power = average_power(x)

    assert power == pytest.approx(
        var + mu ** 2
    )


def test_summary_contains_all_statistics() -> None:
    x = np.array([1.0, 2.0, 3.0])

    summary = summarize_signal_statistics(x)

    assert set(summary) == {
        "mean",
        "variance",
        "rms",
        "average_power",
    }


def test_statistics_reject_empty_signal() -> None:
    x = np.array([], dtype=np.float64)

    with pytest.raises(ValueError):
        mean(x)

    with pytest.raises(ValueError):
        variance(x)

    with pytest.raises(ValueError):
        average_power(x)


def test_statistics_reject_nonfinite_signal() -> None:
    x = np.array(
        [1.0, np.nan, 3.0]
    )

    with pytest.raises(ValueError):
        mean(x)

    with pytest.raises(ValueError):
        variance(x)

    with pytest.raises(ValueError):
        average_power(x)