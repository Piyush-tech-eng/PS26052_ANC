from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConvergenceAnalysisResult:
    """
    Results of LMS convergence analysis.
    """

    learning_curve: np.ndarray
    coefficient_change: np.ndarray
    coefficient_change_norm: np.ndarray
    coefficient_change_curve: np.ndarray

    initial_error_power: float
    final_error_power: float
    error_power_ratio: float

    maximum_coefficient_change: float
    final_coefficient_change: float
    coefficient_change_ratio: float

    error_decreased: bool
    coefficients_stabilized: bool


def moving_average(
    values: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Compute a causal moving average.

    The output has the same length as the input.

    For each index n:

        y[n] =
        mean(
            x[max(0, n-window+1):n+1]
        )
    """

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            "values must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            "values must not be empty."
        )

    if not isinstance(
        window_size,
        int,
    ):
        raise TypeError(
            "window_size must be an integer."
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            "values contains NaN or Inf."
        )

    cumulative_sum = np.cumsum(
        values,
        dtype=np.float64,
    )

    result = np.empty_like(
        values,
        dtype=np.float64,
    )

    for index in range(
        len(values)
    ):
        start_index = max(
            0,
            index - window_size + 1,
        )

        total = cumulative_sum[index]

        if start_index > 0:
            total -= cumulative_sum[
                start_index - 1
            ]

        count = (
            index
            - start_index
            + 1
        )

        result[index] = (
            total / count
        )

    return result


def analyze_lms_convergence(
    error: np.ndarray,
    coefficient_history: np.ndarray,
    *,
    learning_window: int | None = None,
    stability_window: int | None = None,
) -> ConvergenceAnalysisResult:
    """
    Analyze LMS convergence behaviour.

    Parameters
    ----------
    error:
        LMS error signal.

    coefficient_history:
        Matrix with shape:

            (num_samples, filter_length)

        where each row contains the coefficient
        vector after the LMS update for that sample.

    learning_window:
        Window size used to smooth squared error.

    stability_window:
        Window size used to smooth coefficient movement.

    Returns
    -------
    ConvergenceAnalysisResult
        Complete numerical convergence diagnostics.
    """

    error = np.asarray(
        error,
        dtype=np.float64,
    )

    coefficient_history = np.asarray(
        coefficient_history,
        dtype=np.float64,
    )

    if error.ndim != 1:
        raise ValueError(
            "error must be one-dimensional."
        )

    if len(error) == 0:
        raise ValueError(
            "error must not be empty."
        )

    if coefficient_history.ndim != 2:
        raise ValueError(
            "coefficient_history must be "
            "two-dimensional."
        )

    if (
        coefficient_history.shape[0]
        != len(error)
    ):
        raise ValueError(
            "coefficient_history must contain "
            "one row per error sample."
        )

    if coefficient_history.shape[1] == 0:
        raise ValueError(
            "coefficient_history must contain "
            "at least one coefficient."
        )

    if not np.isfinite(
        error
    ).all():
        raise ValueError(
            "error contains NaN or Inf."
        )

    if not np.isfinite(
        coefficient_history
    ).all():
        raise ValueError(
            "coefficient_history contains "
            "NaN or Inf."
        )

    num_samples = len(
        error
    )

    if learning_window is None:
        learning_window = min(
            1000,
            num_samples,
        )

    if stability_window is None:
        stability_window = min(
            1000,
            num_samples,
        )

    if learning_window <= 0:
        raise ValueError(
            "learning_window must be positive."
        )

    if stability_window <= 0:
        raise ValueError(
            "stability_window must be positive."
        )

    squared_error = error ** 2

    learning_curve = moving_average(
        squared_error,
        learning_window,
    )

    coefficient_change = np.empty_like(
        coefficient_history,
        dtype=np.float64,
    )

    coefficient_change[0] = (
        coefficient_history[0]
    )

    if num_samples > 1:
        coefficient_change[1:] = (
            coefficient_history[1:]
            - coefficient_history[:-1]
        )

    coefficient_change_norm = (
        np.linalg.norm(
            coefficient_change,
            axis=1,
        )
    )

    coefficient_change_curve = (
        moving_average(
            coefficient_change_norm,
            stability_window,
        )
    )

    comparison_window = min(
        1000,
        max(
            1,
            num_samples // 4,
        ),
    )

    initial_error_power = float(
        np.mean(
            squared_error[
                :comparison_window
            ]
        )
    )

    final_error_power = float(
        np.mean(
            squared_error[
                -comparison_window:
            ]
        )
    )

    maximum_coefficient_change = float(
        np.max(
            coefficient_change_norm
        )
    )

    final_coefficient_change = float(
        np.mean(
            coefficient_change_norm[
                -comparison_window:
            ]
        )
    )

    epsilon = np.finfo(
        np.float64
    ).eps

    error_power_ratio = float(
        final_error_power
        / (
            initial_error_power
            + epsilon
        )
    )

    coefficient_change_ratio = float(
        final_coefficient_change
        / (
            maximum_coefficient_change
            + epsilon
        )
    )

    error_decreased = bool(
        final_error_power
        < initial_error_power
    )

    coefficients_stabilized = bool(
        final_coefficient_change
        < maximum_coefficient_change
    )

    return ConvergenceAnalysisResult(
        learning_curve=learning_curve,
        coefficient_change=(
            coefficient_change
        ),
        coefficient_change_norm=(
            coefficient_change_norm
        ),
        coefficient_change_curve=(
            coefficient_change_curve
        ),
        initial_error_power=(
            initial_error_power
        ),
        final_error_power=(
            final_error_power
        ),
        error_power_ratio=(
            error_power_ratio
        ),
        maximum_coefficient_change=(
            maximum_coefficient_change
        ),
        final_coefficient_change=(
            final_coefficient_change
        ),
        coefficient_change_ratio=(
            coefficient_change_ratio
        ),
        error_decreased=(
            error_decreased
        ),
        coefficients_stabilized=(
            coefficients_stabilized
        ),
    )