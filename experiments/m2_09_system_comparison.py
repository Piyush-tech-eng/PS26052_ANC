from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.statistics import (
    align_impulse_responses,
    coefficient_correlation,
    coefficient_error,
    coefficient_mse,
    coefficient_rmse,
    relative_coefficient_error,
)


def _load_impulse_response(
    path: Path,
) -> np.ndarray:
    """
    Load and validate a one-dimensional real-valued
    impulse response.
    """

    if not path.is_file():
        raise FileNotFoundError(
            "Ground-truth impulse response file was not found: "
            f"{path}"
        )

    try:
        impulse_response = np.load(
            path,
            allow_pickle=False,
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            "Could not load ground-truth impulse response: "
            f"{path}"
        ) from error

    impulse_response = np.asarray(
        impulse_response,
        dtype=np.float64,
    )

    if impulse_response.ndim != 1:
        raise ValueError(
            "Ground-truth impulse response must be "
            "one-dimensional."
        )

    if len(impulse_response) == 0:
        raise ValueError(
            "Ground-truth impulse response must not be empty."
        )

    if not np.isfinite(impulse_response).all():
        raise ValueError(
            "Ground-truth impulse response contains NaN or Inf."
        )

    return impulse_response


def _load_wiener_coefficients(
    path: Path,
) -> np.ndarray:
    """
    Load and validate the estimated Wiener coefficient vector.
    """

    if not path.is_file():
        raise FileNotFoundError(
            "Wiener coefficient file was not found: "
            f"{path}"
        )

    try:
        coefficients = np.load(
            path,
            allow_pickle=False,
        )
    except (OSError, ValueError) as error:
        raise ValueError(
            "Could not load Wiener coefficients: "
            f"{path}"
        ) from error

    coefficients = np.asarray(
        coefficients,
        dtype=np.float64,
    )

    if coefficients.ndim != 1:
        raise ValueError(
            "Wiener coefficient vector must be one-dimensional."
        )

    if len(coefficients) == 0:
        raise ValueError(
            "Wiener coefficient vector must not be empty."
        )

    if not np.isfinite(coefficients).all():
        raise ValueError(
            "Wiener coefficient vector contains NaN or Inf."
        )

    return coefficients


def main() -> None:

    # -------------------------------------------------
    # Project root
    # -------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    # -------------------------------------------------
    # Input directories
    # -------------------------------------------------

    handoff_directory = (
        project_root
        / "results"
        / "m1_handoff"
    )

    wiener_directory = (
        project_root
        / "results"
        / "m2_06_wiener_solution"
    )

    # -------------------------------------------------
    # Exact input files
    # -------------------------------------------------

    true_impulse_response_path = (
        handoff_directory
        / "system_impulse_response.npy"
    )

    wiener_coefficients_path = (
        wiener_directory
        / "wiener_coefficients.npy"
    )

    # -------------------------------------------------
    # Output directory
    # -------------------------------------------------

    results_directory = (
        project_root
        / "results"
        / "m2_09_system_comparison"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # Validate directories
    # -------------------------------------------------

    if not handoff_directory.is_dir():
        raise FileNotFoundError(
            "Module 1 handoff directory was not found: "
            f"{handoff_directory}"
        )

    if not wiener_directory.is_dir():
        raise FileNotFoundError(
            "M2.06 Wiener solution directory was not found: "
            f"{wiener_directory}"
        )

    # -------------------------------------------------
    # Load true Module 1 FIR system
    #
    # h_true[n]
    # -------------------------------------------------

    h_true = _load_impulse_response(
        true_impulse_response_path
    )

    # -------------------------------------------------
    # Load estimated Wiener coefficients
    #
    # w_opt[n]
    # -------------------------------------------------

    w_opt = _load_wiener_coefficients(
        wiener_coefficients_path
    )

    # -------------------------------------------------
    # Align causal impulse responses
    #
    # The shorter response is zero-padded.
    # -------------------------------------------------

    aligned_true, aligned_estimated = (
        align_impulse_responses(
            h_true,
            w_opt,
        )
    )

    # -------------------------------------------------
    # Coefficient error
    #
    # e_h[n] = h_true[n] - w_opt[n]
    # -------------------------------------------------

    error = coefficient_error(
        h_true,
        w_opt,
    )

    # -------------------------------------------------
    # Metrics
    # -------------------------------------------------

    mse = coefficient_mse(
        h_true,
        w_opt,
    )

    rmse = coefficient_rmse(
        h_true,
        w_opt,
    )

    relative_error = (
        relative_coefficient_error(
            h_true,
            w_opt,
        )
    )

    correlation = (
        coefficient_correlation(
            h_true,
            w_opt,
        )
    )

    true_norm = float(
        np.linalg.norm(
            aligned_true
        )
    )

    estimated_norm = float(
        np.linalg.norm(
            aligned_estimated
        )
    )

    error_norm = float(
        np.linalg.norm(
            error
        )
    )

    maximum_absolute_error = float(
        np.max(
            np.abs(
                error
            )
        )
    )

    # -------------------------------------------------
    # Save numerical artifacts
    # -------------------------------------------------

    np.save(
        results_directory
        / "true_impulse_response.npy",
        aligned_true,
    )

    np.save(
        results_directory
        / "estimated_impulse_response.npy",
        aligned_estimated,
    )

    np.save(
        results_directory
        / "coefficient_error.npy",
        error,
    )

    # -------------------------------------------------
    # Save JSON summary
    # -------------------------------------------------

    summary = {
        "module": "M2.09",
        "purpose": (
            "Compare the Wiener-estimated FIR coefficient "
            "vector with the known Module 1 FIR system."
        ),
        "equation": (
            "h_true compared with w_opt"
        ),
        "input_files": {
            "true_impulse_response": str(
                true_impulse_response_path
            ),
            "wiener_coefficients": str(
                wiener_coefficients_path
            ),
        },
        "true_filter_length": int(
            len(h_true)
        ),
        "estimated_filter_length": int(
            len(w_opt)
        ),
        "comparison_length": int(
            len(aligned_true)
        ),
        "alignment": (
            "Shorter causal impulse response zero-padded "
            "to the length of the longer response."
        ),
        "metrics": {
            "coefficient_mse": mse,
            "coefficient_rmse": rmse,
            "relative_l2_error": (
                relative_error
            ),
            "coefficient_correlation": (
                correlation
            ),
            "true_l2_norm": (
                true_norm
            ),
            "estimated_l2_norm": (
                estimated_norm
            ),
            "error_l2_norm": (
                error_norm
            ),
            "maximum_absolute_coefficient_error": (
                maximum_absolute_error
            ),
        },
        "output_files": {
            "true_impulse_response": (
                "true_impulse_response.npy"
            ),
            "estimated_impulse_response": (
                "estimated_impulse_response.npy"
            ),
            "coefficient_error": (
                "coefficient_error.npy"
            ),
        },
    }

    summary_path = (
        results_directory
        / "system_comparison_summary.json"
    )

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

        file.write("\n")

    # -------------------------------------------------
    # Console report
    # -------------------------------------------------

    print(
        "M2.09 — Wiener System Comparison"
    )

    print(
        "---------------------------------"
    )

    print(
        "Ground-truth impulse response:"
    )

    print(
        true_impulse_response_path
    )

    print()

    print(
        "Estimated Wiener coefficients:"
    )

    print(
        wiener_coefficients_path
    )

    print()

    print(
        f"True filter length: "
        f"{len(h_true)}"
    )

    print(
        f"Estimated filter length: "
        f"{len(w_opt)}"
    )

    print(
        f"Comparison length: "
        f"{len(aligned_true)}"
    )

    print()

    print(
        f"Coefficient MSE: "
        f"{mse:.10g}"
    )

    print(
        f"Coefficient RMSE: "
        f"{rmse:.10g}"
    )

    print(
        f"Relative L2 error: "
        f"{relative_error:.10g}"
    )

    print(
        f"Coefficient correlation: "
        f"{correlation:.10g}"
    )

    print()

    print(
        f"True filter L2 norm: "
        f"{true_norm:.10g}"
    )

    print(
        f"Estimated filter L2 norm: "
        f"{estimated_norm:.10g}"
    )

    print(
        f"Error L2 norm: "
        f"{error_norm:.10g}"
    )

    print(
        "Maximum absolute coefficient error: "
        f"{maximum_absolute_error:.10g}"
    )

    print()

    print(
        f"Results saved to: "
        f"{results_directory}"
    )


if __name__ == "__main__":
    main()