from __future__ import annotations

import numpy as np


def autocorrelation(
    signal: np.ndarray,
    max_lag: int | None = None,
    *,
    unbiased: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the autocorrelation of a real-valued finite signal.

    Parameters
    ----------
    signal:
        One-dimensional real-valued signal x[n].

    max_lag:
        Maximum non-negative lag to compute.
        Defaults to N - 1.

    unbiased:
        If False, use the biased estimator:

            R[k] = (1/N) *
                   sum_{n=k}^{N-1} x[n]x[n-k]

        If True, use the unbiased estimator:

            R[k] = (1/(N-k)) *
                   sum_{n=k}^{N-1} x[n]x[n-k]

    Returns
    -------
    lags:
        Array [0, 1, ..., max_lag].

    values:
        Corresponding autocorrelation estimates.
    """

    samples = np.asarray(
        signal,
        dtype=np.float64,
    )

    if samples.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if len(samples) == 0:
        raise ValueError(
            "Signal must not be empty."
        )

    if not np.isfinite(samples).all():
        raise ValueError(
            "Signal contains NaN or Inf."
        )

    num_samples = len(samples)

    if max_lag is None:
        max_lag = num_samples - 1

    if not isinstance(max_lag, (int, np.integer)):
        raise ValueError(
            "max_lag must be an integer."
        )

    if max_lag < 0:
        raise ValueError(
            "max_lag must be non-negative."
        )

    if max_lag >= num_samples:
        raise ValueError(
            "max_lag must be smaller than the number of samples."
        )

    lags = np.arange(
        max_lag + 1,
        dtype=np.int64,
    )

    values = np.empty(
        max_lag + 1,
        dtype=np.float64,
    )

    for index, lag in enumerate(lags):
        overlap = samples[lag:] * samples[:num_samples - lag]

        denominator = (
            num_samples - lag
            if unbiased
            else num_samples
        )

        values[index] = (
            np.sum(overlap) / denominator
        )

    return lags, values

def cross_correlation(
    reference: np.ndarray,
    desired: np.ndarray,
    max_lag: int | None = None,
    *,
    unbiased: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate the cross-correlation between reference x[n]
    and desired signal d[n].

    The convention used is:

        Rxd[k] = (1 / N)
                 * sum_{n=k}^{N-1}
                   d[n] x[n-k]

    for the biased estimator.

    This convention is chosen so that the resulting sequence
    can directly form the Wiener cross-correlation vector:

        p = [
            Rxd[0],
            Rxd[1],
            ...
        ]^T

    Parameters
    ----------
    reference:
        One-dimensional real-valued reference signal x[n].

    desired:
        One-dimensional real-valued desired signal d[n].

    max_lag:
        Maximum non-negative lag to compute.
        Defaults to N - 1.

    unbiased:
        If False, use the biased estimator:

            Rxd[k] = (1 / N)
                     * sum d[n] x[n-k]

        If True, use the overlap-normalized estimator:

            Rxd[k] = (1 / (N-k))
                     * sum d[n] x[n-k]

    Returns
    -------
    lags:
        Array [0, 1, ..., max_lag].

    values:
        Corresponding cross-correlation estimates.
    """

    x = np.asarray(
        reference,
        dtype=np.float64,
    )

    d = np.asarray(
        desired,
        dtype=np.float64,
    )

    if x.ndim != 1:
        raise ValueError(
            "Reference signal must be one-dimensional."
        )

    if d.ndim != 1:
        raise ValueError(
            "Desired signal must be one-dimensional."
        )

    if len(x) == 0:
        raise ValueError(
            "Reference signal must not be empty."
        )

    if len(d) == 0:
        raise ValueError(
            "Desired signal must not be empty."
        )

    if len(x) != len(d):
        raise ValueError(
            "Reference and desired signals must have "
            "equal lengths."
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "Reference signal contains NaN or Inf."
        )

    if not np.isfinite(d).all():
        raise ValueError(
            "Desired signal contains NaN or Inf."
        )

    num_samples = len(x)

    if max_lag is None:
        max_lag = num_samples - 1

    if not isinstance(
        max_lag,
        (int, np.integer),
    ):
        raise ValueError(
            "max_lag must be an integer."
        )

    if max_lag < 0:
        raise ValueError(
            "max_lag must be non-negative."
        )

    if max_lag >= num_samples:
        raise ValueError(
            "max_lag must be smaller than the number "
            "of samples."
        )

    lags = np.arange(
        max_lag + 1,
        dtype=np.int64,
    )

    values = np.empty(
        max_lag + 1,
        dtype=np.float64,
    )

    for index, lag in enumerate(lags):

        # d[n] x[n-k]
        overlap = (
            d[lag:]
            * x[:num_samples - lag]
        )

        denominator = (
            num_samples - lag
            if unbiased
            else num_samples
        )

        values[index] = (
            np.sum(overlap)
            / denominator
        )

    return lags, values

