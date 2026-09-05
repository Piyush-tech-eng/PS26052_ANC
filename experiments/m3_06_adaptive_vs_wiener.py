from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from anc.adaptive import (
    compare_against_wiener,
)


def load_array(
    path: Path,
    *,
    name: str,
) -> np.ndarray:
    """
    Load and validate a NumPy array.
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

    if not np.isfinite(
        values
    ).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return values


def save_comparison_results(
    *,
    algorithm_name: str,
    results_directory: Path,
    comparison,
) -> dict[str, object]:
    """
    Save numerical Wiener-comparison artifacts for
    one adaptive algorithm.
    """

    algorithm_directory = (
        results_directory
        / algorithm_name
    )

    algorithm_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        algorithm_directory
        / "coefficient_error.npy",
        comparison.coefficient_error,
    )

    np.save(
        algorithm_directory
        / "coefficient_error_norm.npy",
        comparison.coefficient_error_norm,
    )

    return {
        "initial_coefficient_error": (
            comparison.initial_coefficient_error
        ),
        "final_coefficient_error": (
            comparison.final_coefficient_error
        ),
        "coefficient_error_ratio": (
            comparison.coefficient_error_ratio
        ),
        "minimum_coefficient_error": (
            comparison.minimum_coefficient_error
        ),
        "minimum_error_index": (
            comparison.minimum_error_index
        ),
        "moved_closer_to_wiener": (
            comparison.moved_closer_to_wiener
        ),
    }


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Compare LMS and NLMS coefficient "
            "trajectories against Wiener coefficients."
        )
    )

    parser.add_argument(
        "--wiener-coefficients",
        type=Path,
        required=True,
        help=(
            "Path to the Module 2 Wiener coefficient "
            ".npy file."
        ),
    )

    arguments = parser.parse_args()

    project_root = Path(
        __file__
    ).resolve().parents[1]

    adaptive_results_directory = (
        project_root
        / "results"
        / "m3_05_lms_vs_nlms"
    )

    results_directory = (
        project_root
        / "results"
        / "m3_06_adaptive_vs_wiener"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------
    # Load Wiener solution
    # ---------------------------------------------

    wiener_coefficients = load_array(
        arguments.wiener_coefficients,
        name="Wiener coefficients",
    )

    if wiener_coefficients.ndim != 1:
        raise ValueError(
            "Wiener coefficients must be "
            "one-dimensional."
        )

    # ---------------------------------------------
    # Load LMS trajectory
    # ---------------------------------------------

    lms_history_path = (
        adaptive_results_directory
        / "lms"
        / "coefficient_history.npy"
    )

    lms_coefficient_history = load_array(
        lms_history_path,
        name="LMS coefficient history",
    )

    if lms_coefficient_history.ndim != 2:
        raise ValueError(
            "LMS coefficient history must be "
            "two-dimensional."
        )

    # ---------------------------------------------
    # Load NLMS trajectory
    # ---------------------------------------------

    nlms_history_path = (
        adaptive_results_directory
        / "nlms"
        / "coefficient_history.npy"
    )

    nlms_coefficient_history = load_array(
        nlms_history_path,
        name="NLMS coefficient history",
    )

    if nlms_coefficient_history.ndim != 2:
        raise ValueError(
            "NLMS coefficient history must be "
            "two-dimensional."
        )

    # ---------------------------------------------
    # Compare LMS with Wiener solution
    # ---------------------------------------------

    print(
        "Comparing LMS against Wiener solution..."
    )

    lms_comparison = (
        compare_against_wiener(
            lms_coefficient_history,
            wiener_coefficients,
        )
    )

    # ---------------------------------------------
    # Compare NLMS with Wiener solution
    # ---------------------------------------------

    print(
        "Comparing NLMS against Wiener solution..."
    )

    nlms_comparison = (
        compare_against_wiener(
            nlms_coefficient_history,
            wiener_coefficients,
        )
    )

    # ---------------------------------------------
    # Save numerical artifacts
    # ---------------------------------------------

    lms_summary = (
        save_comparison_results(
            algorithm_name="lms",
            results_directory=results_directory,
            comparison=lms_comparison,
        )
    )

    nlms_summary = (
        save_comparison_results(
            algorithm_name="nlms",
            results_directory=results_directory,
            comparison=nlms_comparison,
        )
    )

    # ---------------------------------------------
    # Determine which algorithm finished closer
    # ---------------------------------------------

    if (
        lms_comparison.final_coefficient_error
        < nlms_comparison.final_coefficient_error
    ):

        closer_algorithm = "LMS"

    elif (
        nlms_comparison.final_coefficient_error
        < lms_comparison.final_coefficient_error
    ):

        closer_algorithm = "NLMS"

    else:

        closer_algorithm = "Tie"

    # ---------------------------------------------
    # Save summary
    # ---------------------------------------------

    summary = {
        "module": "M3.06",
        "experiment": (
            "Adaptive coefficient convergence "
            "against Wiener benchmark"
        ),
        "wiener_coefficients_path": str(
            arguments.wiener_coefficients.resolve()
        ),
        "filter_length": int(
            len(wiener_coefficients)
        ),
        "lms": lms_summary,
        "nlms": nlms_summary,
        "final_closer_algorithm": (
            closer_algorithm
        ),
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
    # Console output
    # ---------------------------------------------

    print()

    print(
        "M3.06 Adaptive vs Wiener comparison "
        "completed."
    )

    print()

    print(
        "LMS:"
    )

    print(
        f"  Initial coefficient error: "
        f"{lms_comparison.initial_coefficient_error:.10g}"
    )

    print(
        f"  Final coefficient error: "
        f"{lms_comparison.final_coefficient_error:.10g}"
    )

    print(
        f"  Error ratio: "
        f"{lms_comparison.coefficient_error_ratio:.10g}"
    )

    print(
        f"  Minimum coefficient error: "
        f"{lms_comparison.minimum_coefficient_error:.10g}"
    )

    print(
        f"  Moved closer to Wiener: "
        f"{lms_comparison.moved_closer_to_wiener}"
    )

    print()

    print(
        "NLMS:"
    )

    print(
        f"  Initial coefficient error: "
        f"{nlms_comparison.initial_coefficient_error:.10g}"
    )

    print(
        f"  Final coefficient error: "
        f"{nlms_comparison.final_coefficient_error:.10g}"
    )

    print(
        f"  Error ratio: "
        f"{nlms_comparison.coefficient_error_ratio:.10g}"
    )

    print(
        f"  Minimum coefficient error: "
        f"{nlms_comparison.minimum_coefficient_error:.10g}"
    )

    print(
        f"  Moved closer to Wiener: "
        f"{nlms_comparison.moved_closer_to_wiener}"
    )

    print()

    print(
        f"Algorithm finishing closer to "
        f"Wiener solution: "
        f"{closer_algorithm}"
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