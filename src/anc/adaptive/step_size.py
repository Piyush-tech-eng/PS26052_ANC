from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from anc.adaptive.lms import LMSFilter


@dataclass(frozen=True)
class StepSizeRunResult:
    """
    Result from one LMS step-size run.
    """

    step_size: float
    stable: bool
    converged: bool

    mse: float | None
    rmse: float | None

    initial_error_power: float | None
    final_error_power: float | None
    error_power_ratio: float | None

    final_coefficient_norm: float | None

    reason: str | None


def run_lms_step_size(
    reference: np.ndarray,
    desired: np.ndarray,
    *,
    filter_length: int,
    step_size: float,
) -> StepSizeRunResult:
    """
    Run LMS with one step size and evaluate
    its stability and convergence.
    """

    reference = np.asarray(
        reference,
        dtype=np.float64,
    )

    desired = np.asarray(
        desired,
        dtype=np.float64,
    )

    try:

        with np.errstate(
            over="raise",
            invalid="raise",
        ):

            lms = LMSFilter(
                filter_length=filter_length,
                step_size=step_size,
            )

            result = lms.adapt(
                reference,
                desired,
            )

            error = result.error

            if not np.isfinite(
                error
            ).all():
                raise FloatingPointError(
                    "Error became non-finite."
                )

            if not np.isfinite(
                result.final_coefficients
            ).all():
                raise FloatingPointError(
                    "Coefficients became non-finite."
                )

            squared_error = error ** 2

            num_samples = len(
                error
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

            mse = float(
                np.mean(
                    squared_error
                )
            )

            rmse = float(
                np.sqrt(
                    mse
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

            converged = bool(
                final_error_power
                < initial_error_power
            )

            return StepSizeRunResult(
                step_size=float(
                    step_size
                ),
                stable=True,
                converged=converged,
                mse=mse,
                rmse=rmse,
                initial_error_power=(
                    initial_error_power
                ),
                final_error_power=(
                    final_error_power
                ),
                error_power_ratio=(
                    error_power_ratio
                ),
                final_coefficient_norm=(
                    float(
                        np.linalg.norm(
                            result.final_coefficients
                        )
                    )
                ),
                reason=None,
            )

    except (
        FloatingPointError,
        ValueError,
    ) as error:

        return StepSizeRunResult(
            step_size=float(
                step_size
            ),
            stable=False,
            converged=False,
            mse=None,
            rmse=None,
            initial_error_power=None,
            final_error_power=None,
            error_power_ratio=None,
            final_coefficient_norm=None,
            reason=str(
                error
            ),
        )


def run_lms_step_size_sweep(
    reference: np.ndarray,
    desired: np.ndarray,
    *,
    filter_length: int,
    step_sizes: list[float],
) -> list[StepSizeRunResult]:
    """
    Run LMS across multiple step sizes.
    """

    if len(step_sizes) == 0:
        raise ValueError(
            "step_sizes must not be empty."
        )

    results = []

    for step_size in step_sizes:

        result = run_lms_step_size(
            reference,
            desired,
            filter_length=filter_length,
            step_size=step_size,
        )

        results.append(
            result
        )

    return results