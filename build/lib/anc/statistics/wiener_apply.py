from __future__ import annotations

import numpy as np


def apply_fir(
    signal: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """
    Apply a causal FIR filter:

        y[n] = sum_k h[k] x[n-k]

    with zero initial conditions.
    """

    x = np.asarray(signal, dtype=np.float64)
    h = np.asarray(coefficients, dtype=np.float64)

    if x.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if h.ndim != 1:
        raise ValueError(
            "FIR coefficients must be one-dimensional."
        )

    if len(x) == 0:
        raise ValueError(
            "Signal must not be empty."
        )

    if len(h) == 0:
        raise ValueError(
            "FIR coefficients must not be empty."
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "Signal contains NaN or Inf."
        )

    if not np.isfinite(h).all():
        raise ValueError(
            "FIR coefficients contain NaN or Inf."
        )

    # Full causal convolution contains the filter tail.
    # We retain exactly len(x) samples, matching the
    # convention used by the project's FIR implementation.
    output = np.convolve(
        x,
        h,
        mode="full",
    )[:len(x)]

    return np.asarray(
        output,
        dtype=np.float64,
    )


def mean_squared_error(
    target: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """
    Compute:

        MSE = (1/N) sum_n (target[n] - estimate[n])^2
    """

    target_array = np.asarray(
        target,
        dtype=np.float64,
    )

    estimate_array = np.asarray(
        estimate,
        dtype=np.float64,
    )

    if target_array.ndim != 1:
        raise ValueError(
            "Target must be one-dimensional."
        )

    if estimate_array.ndim != 1:
        raise ValueError(
            "Estimate must be one-dimensional."
        )

    if len(target_array) != len(estimate_array):
        raise ValueError(
            "Target and estimate must have equal lengths."
        )

    if len(target_array) == 0:
        raise ValueError(
            "Target and estimate must not be empty."
        )

    if not np.isfinite(target_array).all():
        raise ValueError(
            "Target contains NaN or Inf."
        )

    if not np.isfinite(estimate_array).all():
        raise ValueError(
            "Estimate contains NaN or Inf."
        )

    error = (
        target_array
        - estimate_array
    )

    return float(
        np.mean(error ** 2)
    )


def root_mean_squared_error(
    target: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """Compute the square root of mean squared error."""

    return float(
        np.sqrt(
            mean_squared_error(
                target,
                estimate,
            )
        )
    )


def normalized_correlation(
    target: np.ndarray,
    estimate: np.ndarray,
) -> float:
    """
    Compute normalized zero-mean correlation:

        corr =
        sum((d-mu_d)(dh-mu_dh))
        --------------------------------
        sqrt(sum((d-mu_d)^2) sum((dh-mu_dh)^2))

    Returns a value in [-1, 1].
    """

    target_array = np.asarray(
        target,
        dtype=np.float64,
    )

    estimate_array = np.asarray(
        estimate,
        dtype=np.float64,
    )

    if target_array.ndim != 1:
        raise ValueError(
            "Target must be one-dimensional."
        )

    if estimate_array.ndim != 1:
        raise ValueError(
            "Estimate must be one-dimensional."
        )

    if len(target_array) != len(estimate_array):
        raise ValueError(
            "Target and estimate must have equal lengths."
        )

    if len(target_array) == 0:
        raise ValueError(
            "Target and estimate must not be empty."
        )

    if not np.isfinite(target_array).all():
        raise ValueError(
            "Target contains NaN or Inf."
        )

    if not np.isfinite(estimate_array).all():
        raise ValueError(
            "Estimate contains NaN or Inf."
        )

    target_centered = (
        target_array
        - np.mean(target_array)
    )

    estimate_centered = (
        estimate_array
        - np.mean(estimate_array)
    )

    denominator = (
        np.linalg.norm(target_centered)
        * np.linalg.norm(estimate_centered)
    )

    if denominator == 0.0:
        raise ValueError(
            "Correlation is undefined when either "
            "signal has zero variance."
        )

    return float(
        np.dot(
            target_centered,
            estimate_centered,
        )
        / denominator
    )