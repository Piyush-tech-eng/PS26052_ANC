from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.statistics import (
    solve_wiener_hopf,
)


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    # -----------------------------------------
    # Input: M2.05 artifacts
    # -----------------------------------------

    input_directory = (
        project_root
        / "results"
        / "m2_05_wiener_structures"
    )

    correlation_matrix_path = (
        input_directory
        / "correlation_matrix.npy"
    )

    cross_correlation_vector_path = (
        input_directory
        / "cross_correlation_vector.npy"
    )

    # -----------------------------------------
    # Output: M2.06 artifacts
    # -----------------------------------------

    results_directory = (
        project_root
        / "results"
        / "m2_06_wiener_solution"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # Load M2.05 Wiener system
    # -----------------------------------------

    R = np.load(
        correlation_matrix_path,
        allow_pickle=False,
    )

    p = np.load(
        cross_correlation_vector_path,
        allow_pickle=False,
    )

    # -----------------------------------------
    # Solve Wiener-Hopf equation
    #
    #     R @ w_opt = p
    # -----------------------------------------

    w_opt = solve_wiener_hopf(
        R,
        p,
    )

    # -----------------------------------------
    # Numerical validation
    # -----------------------------------------

    reconstructed_p = R @ w_opt

    residual = (
        reconstructed_p - p
    )

    residual_norm = float(
        np.linalg.norm(residual)
    )

    relative_residual = float(
        residual_norm
        / (
            np.linalg.norm(p)
            + np.finfo(np.float64).eps
        )
    )

    condition_number = float(
        np.linalg.cond(R)
    )

    solution_is_finite = bool(
        np.isfinite(w_opt).all()
    )

    # -----------------------------------------
    # Save coefficient vector
    # -----------------------------------------

    np.save(
        results_directory
        / "wiener_coefficients.npy",
        w_opt,
    )

    # -----------------------------------------
    # Save summary
    # -----------------------------------------

    summary = {
        "module": "M2.06",
        "equation": "R @ w_opt = p",
        "filter_length": int(
            len(w_opt)
        ),
        "solution": {
            "finite": solution_is_finite,
            "minimum_coefficient": float(
                np.min(w_opt)
            ),
            "maximum_coefficient": float(
                np.max(w_opt)
            ),
            "l2_norm": float(
                np.linalg.norm(w_opt)
            ),
        },
        "numerical_validation": {
            "residual_l2_norm": residual_norm,
            "relative_residual": relative_residual,
            "condition_number": condition_number,
        },
        "input_files": {
            "correlation_matrix": (
                "m2_05_wiener_structures/"
                "correlation_matrix.npy"
            ),
            "cross_correlation_vector": (
                "m2_05_wiener_structures/"
                "cross_correlation_vector.npy"
            ),
        },
        "output_files": {
            "wiener_coefficients": (
                "wiener_coefficients.npy"
            ),
        },
    }

    summary_path = (
        results_directory
        / "wiener_solution_summary.json"
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

    # -----------------------------------------
    # Report
    # -----------------------------------------

    print(
        "M2.06 — Wiener-Hopf Solution"
    )

    print(
        "-----------------------------"
    )

    print(
        f"Filter length: {len(w_opt)}"
    )

    print(
        f"Condition number: "
        f"{condition_number:.10g}"
    )

    print(
        f"Residual L2 norm: "
        f"{residual_norm:.10g}"
    )

    print(
        f"Relative residual: "
        f"{relative_residual:.10g}"
    )

    print(
        f"Solution finite: "
        f"{solution_is_finite}"
    )

    print(
        f"Results saved to: "
        f"{results_directory}"
    )


if __name__ == "__main__":
    main()