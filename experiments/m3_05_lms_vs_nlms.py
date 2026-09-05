from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.adaptive import (
    LMSFilter,
    NLMSFilter,
)


def load_signal(
    path: Path,
    *,
    name: str,
) -> np.ndarray:
    """
    Load and validate a one-dimensional signal.
    """

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} was not found: {path}"
        )

    values = np.load(
        path,
        allow_pickle=False,
    )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional."
        )

    if len(values) == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return values


def compute_error_metrics(
    squared_error: np.ndarray,
) -> dict[str, float]:
    """
    Compute comparable learning metrics.
    """

    squared_error = np.asarray(
        squared_error,
        dtype=np.float64,
    )

    num_samples = len(
        squared_error
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

    return {
        "mse": mse,
        "initial_error_power": (
            initial_error_power
        ),
        "final_error_power": (
            final_error_power
        ),
        "error_power_ratio": (
            error_power_ratio
        ),
    }


def main() -> None:

    project_root = Path(
        __file__
    ).resolve().parents[1]

    handoff_directory = (
        project_root
        / "results"
        / "m1_handoff"
    )

    results_directory = (
        project_root
        / "results"
        / "m3_05_lms_vs_nlms"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------
    # Load identical data for both algorithms
    # ---------------------------------------------

    reference = load_signal(
        handoff_directory
        / "reference.npy",
        name="Reference signal",
    )

    desired = load_signal(
        handoff_directory
        / "desired.npy",
        name="Desired signal",
    )

    if len(reference) != len(
        desired
    ):
        raise ValueError(
            "Reference and desired must have "
            "the same length."
        )

    # ---------------------------------------------
    # Experiment configuration
    # ---------------------------------------------

    filter_length = 32

    # Use the previously validated LMS
    # operating step size.
    lms_step_size = 1e-3

    # NLMS step size is normalized and therefore
    # is not directly equivalent numerically to LMS.
    nlms_step_size = 0.1

    epsilon = 1e-8

    # ---------------------------------------------
    # Run LMS
    # ---------------------------------------------

    print(
        "Running LMS..."
    )

    lms = LMSFilter(
        filter_length=filter_length,
        step_size=lms_step_size,
    )

    lms_result = lms.adapt(
        reference,
        desired,
    )

    # ---------------------------------------------
    # Run NLMS
    # ---------------------------------------------

    print(
        "Running NLMS..."
    )

    nlms = NLMSFilter(
        filter_length=filter_length,
        step_size=nlms_step_size,
        epsilon=epsilon,
    )

    nlms_result = nlms.adapt(
        reference,
        desired,
    )

    # ---------------------------------------------
    # Metrics
    # ---------------------------------------------

    lms_squared_error = (
        lms_result.error ** 2
    )

    lms_metrics = (
        compute_error_metrics(
            lms_squared_error
        )
    )

    nlms_metrics = (
        compute_error_metrics(
            nlms_result.squared_error
        )
    )

    # ---------------------------------------------
    # Save LMS artifacts
    # ---------------------------------------------

    lms_directory = (
        results_directory
        / "lms"
    )

    lms_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        lms_directory
        / "output.npy",
        lms_result.output,
    )

    np.save(
        lms_directory
        / "error.npy",
        lms_result.error,
    )

    np.save(
        lms_directory
        / "squared_error.npy",
        lms_squared_error,
    )

    np.save(
        lms_directory
        / "coefficient_history.npy",
        lms_result.coefficient_history,
    )

    np.save(
        lms_directory
        / "final_coefficients.npy",
        lms_result.final_coefficients,
    )

    # ---------------------------------------------
    # Save NLMS artifacts
    # ---------------------------------------------

    nlms_directory = (
        results_directory
        / "nlms"
    )

    nlms_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        nlms_directory
        / "output.npy",
        nlms_result.output,
    )

    np.save(
        nlms_directory
        / "error.npy",
        nlms_result.error,
    )

    np.save(
        nlms_directory
        / "squared_error.npy",
        nlms_result.squared_error,
    )

    np.save(
        nlms_directory
        / "coefficient_history.npy",
        nlms_result.coefficient_history,
    )

    np.save(
        nlms_directory
        / "final_coefficients.npy",
        nlms_result.final_coefficients,
    )

    # ---------------------------------------------
    # Build comparison summary
    # ---------------------------------------------

    summary = {
        "module": "M3.05",
        "experiment": (
            "LMS versus NLMS comparison"
        ),
        "num_samples": int(
            len(reference)
        ),
        "filter_length": (
            filter_length
        ),
        "lms": {
            "step_size": (
                lms_step_size
            ),
            **lms_metrics,
            "final_coefficient_norm": (
                float(
                    np.linalg.norm(
                        lms_result.final_coefficients
                    )
                )
            ),
        },
        "nlms": {
            "step_size": (
                nlms_step_size
            ),
            "epsilon": epsilon,
            **nlms_metrics,
            "final_coefficient_norm": (
                float(
                    np.linalg.norm(
                        nlms_result.final_coefficients
                    )
                )
            ),
        },
    }

    summary_path = (
        results_directory
        / "comparison_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    # ---------------------------------------------
    # Console comparison
    # ---------------------------------------------

    print()

    print(
        "M3.05 LMS vs NLMS comparison "
        "completed."
    )

    print()

    print(
        "LMS:"
    )

    print(
        f"  MSE: "
        f"{lms_metrics['mse']:.10g}"
    )

    print(
        f"  Initial error power: "
        f"{lms_metrics['initial_error_power']:.10g}"
    )

    print(
        f"  Final error power: "
        f"{lms_metrics['final_error_power']:.10g}"
    )

    print(
        f"  Error power ratio: "
        f"{lms_metrics['error_power_ratio']:.10g}"
    )

    print()

    print(
        "NLMS:"
    )

    print(
        f"  MSE: "
        f"{nlms_metrics['mse']:.10g}"
    )

    print(
        f"  Initial error power: "
        f"{nlms_metrics['initial_error_power']:.10g}"
    )

    print(
        f"  Final error power: "
        f"{nlms_metrics['final_error_power']:.10g}"
    )

    print(
        f"  Error power ratio: "
        f"{nlms_metrics['error_power_ratio']:.10g}"
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        results_directory
    )


if __name__ == "__main__":
    main()