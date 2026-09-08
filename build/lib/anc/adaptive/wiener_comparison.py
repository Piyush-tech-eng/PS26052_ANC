from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WienerComparisonResult:
    """
    Results of comparing an adaptive filter trajectory
    against a Wiener-optimal coefficient vector.
    """

    coefficient_error: np.ndarray
    coefficient_error_norm: np.ndarray

    initial_coefficient_error: float
    final_coefficient_error: float
    coefficient_error_ratio: float

    minimum_coefficient_error: float
    minimum_error_index: int

    final_coefficients_match_dimension: bool
    moved_closer_to_wiener: bool


def compare_against_wiener(
    coefficient_history: np.ndarray,
    wiener_coefficients: np.ndarray,
    *,
    comparison_window: int | None = None,
) -> WienerComparisonResult:
    """
    Compare an adaptive coefficient trajectory against
    a Wiener-optimal coefficient vector.

    Parameters
    ----------
    coefficient_history:
        Adaptive filter coefficient history with shape:

            (num_samples, filter_length)

        Each row represents the adaptive coefficient
        vector at one point in time.

    wiener_coefficients:
        Wiener-optimal coefficient vector with shape:

            (filter_length,)

    comparison_window:
        Number of samples used from the beginning and
        end of the trajectory for initial/final
        convergence comparison.

        If None, one quarter of the available samples is
        used, limited to at most 1000 samples.

    Returns
    -------
    WienerComparisonResult
        Numerical diagnostics describing whether the
        adaptive coefficients moved toward the Wiener
        solution.
    """

    coefficient_history = np.asarray(
        coefficient_history,
        dtype=np.float64,
    )

    wiener_coefficients = np.asarray(
        wiener_coefficients,
        dtype=np.float64,
    )

    # ---------------------------------------------
    # Validate coefficient history
    # ---------------------------------------------

    if coefficient_history.ndim != 2:
        raise ValueError(
            "coefficient_history must be "
            "two-dimensional."
        )

    if coefficient_history.shape[0] == 0:
        raise ValueError(
            "coefficient_history must contain "
            "at least one sample."
        )

    if coefficient_history.shape[1] == 0:
        raise ValueError(
            "coefficient_history must contain "
            "at least one coefficient."
        )

    if not np.isfinite(
        coefficient_history
    ).all():
        raise ValueError(
            "coefficient_history contains NaN or Inf."
        )

    # ---------------------------------------------
    # Validate Wiener coefficients
    # ---------------------------------------------

    if wiener_coefficients.ndim != 1:
        raise ValueError(
            "wiener_coefficients must be "
            "one-dimensional."
        )

    if len(wiener_coefficients) == 0:
        raise ValueError(
            "wiener_coefficients must not be empty."
        )

    if not np.isfinite(
        wiener_coefficients
    ).all():
        raise ValueError(
            "wiener_coefficients contains NaN or Inf."
        )

    # ---------------------------------------------
    # Dimension compatibility
    # ---------------------------------------------

    filter_length = (
        coefficient_history.shape[1]
    )

    if len(
        wiener_coefficients
    ) != filter_length:
        raise ValueError(
            "coefficient_history and "
            "wiener_coefficients must have the "
            "same filter length."
        )

    # ---------------------------------------------
    # Comparison window
    # ---------------------------------------------

    num_samples = (
        coefficient_history.shape[0]
    )

    if comparison_window is None:

        comparison_window = min(
            1000,
            max(
                1,
                num_samples // 4,
            ),
        )

    if not isinstance(
        comparison_window,
        int,
    ):
        raise TypeError(
            "comparison_window must be an integer."
        )

    if comparison_window <= 0:
        raise ValueError(
            "comparison_window must be positive."
        )

    comparison_window = min(
        comparison_window,
        num_samples,
    )

    # ---------------------------------------------
    # Coefficient error
    #
    # delta_w[n] =
    #
    # w_adaptive[n] - w_wiener
    # ---------------------------------------------

    coefficient_error = (
        coefficient_history
        - wiener_coefficients
    )

    # ---------------------------------------------
    # Euclidean distance to Wiener solution
    #
    # ||w[n] - w_opt||_2
    # ---------------------------------------------

    coefficient_error_norm = (
        np.linalg.norm(
            coefficient_error,
            axis=1,
        )
    )

    # ---------------------------------------------
    # Compare beginning and end of adaptation
    # ---------------------------------------------

    initial_coefficient_error = float(
        np.mean(
            coefficient_error_norm[
                :comparison_window
            ]
        )
    )

    final_coefficient_error = float(
        np.mean(
            coefficient_error_norm[
                -comparison_window:
            ]
        )
    )

    epsilon = np.finfo(
        np.float64
    ).eps

    coefficient_error_ratio = float(
        final_coefficient_error
        / (
            initial_coefficient_error
            + epsilon
        )
    )

    # ---------------------------------------------
    # Best point reached during adaptation
    # ---------------------------------------------

    minimum_error_index = int(
        np.argmin(
            coefficient_error_norm
        )
    )

    minimum_coefficient_error = float(
        coefficient_error_norm[
            minimum_error_index
        ]
    )

    # ---------------------------------------------
    # Convergence decision
    # ---------------------------------------------

    moved_closer_to_wiener = bool(
        final_coefficient_error
        < initial_coefficient_error
    )

    return WienerComparisonResult(
        coefficient_error=(
            coefficient_error
        ),
        coefficient_error_norm=(
            coefficient_error_norm
        ),
        initial_coefficient_error=(
            initial_coefficient_error
        ),
        final_coefficient_error=(
            final_coefficient_error
        ),
        coefficient_error_ratio=(
            coefficient_error_ratio
        ),
        minimum_coefficient_error=(
            minimum_coefficient_error
        ),
        minimum_error_index=(
            minimum_error_index
        ),
        final_coefficients_match_dimension=True,
        moved_closer_to_wiener=(
            moved_closer_to_wiener
        ),
    )