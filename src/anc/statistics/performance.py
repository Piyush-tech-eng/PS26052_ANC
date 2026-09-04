from __future__ import annotations

import numpy as np


def signal_power(
    signal: np.ndarray,
) -> float:
    """
    Compute finite-signal average power:

        P = (1/N) sum x[n]^2
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

    return float(
        np.mean(samples ** 2)
    )


def orthogonality_residual(
    error: np.ndarray,
    reference: np.ndarray,
    max_lag: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate:

        R_ex[k] =
            (1/N) sum e[n] x[n-k]

    for non-negative lags.

    For the Wiener optimum, these values should
    approach zero.
    """

    e = np.asarray(
        error,
        dtype=np.float64,
    )

    x = np.asarray(
        reference,
        dtype=np.float64,
    )

    if e.ndim != 1:
        raise ValueError(
            "Error must be one-dimensional."
        )

    if x.ndim != 1:
        raise ValueError(
            "Reference must be one-dimensional."
        )

    if len(e) == 0:
        raise ValueError(
            "Error must not be empty."
        )

    if len(x) == 0:
        raise ValueError(
            "Reference must not be empty."
        )

    if len(e) != len(x):
        raise ValueError(
            "Error and reference must have equal lengths."
        )

    if not np.isfinite(e).all():
        raise ValueError(
            "Error contains NaN or Inf."
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "Reference contains NaN or Inf."
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

        overlap = (
            e[lag:]
            * x[:num_samples - lag]
        )

        values[index] = (
            np.sum(overlap)
            / num_samples
        )

    return lags, values


def relative_residual_power(
    target: np.ndarray,
    error: np.ndarray,
) -> float:
    """
    Compute residual power relative to target power:

        P_e / P_d
    """

    target_power = signal_power(target)
    error_power = signal_power(error)

    denominator = (
        target_power
        + np.finfo(np.float64).eps
    )

    return float(
        error_power / denominator
    )


def explained_power_fraction(
    target: np.ndarray,
    error: np.ndarray,
) -> float:
    """
    Compute the fraction of target power not present
    in the residual:

        1 - P_e / P_d
    """

    return float(
        1.0
        - relative_residual_power(
            target,
            error,
        )
    )