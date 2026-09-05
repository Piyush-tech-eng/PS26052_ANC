from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.adaptive import (
    analyze_lms_convergence,
)


def load_array(
    path: Path,
    *,
    name: str,
) -> np.ndarray:

    if not path.is_file():
        raise FileNotFoundError(
            f"{name} was not found: {path}"
        )

    array = np.load(
        path,
        allow_pickle=False,
    )

    return np.asarray(
        array,
        dtype=np.float64,
    )


def main() -> None:

    project_root = Path(
        __file__
    ).resolve().parents[1]

    input_directory = (
        project_root
        / "results"
        / "m3_02_lms"
    )

    results_directory = (
        project_root
        / "results"
        / "m3_03_lms_convergence"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    error = load_array(
        input_directory
        / "error.npy",
        name="LMS error",
    )

    coefficient_history = load_array(
        input_directory
        / "coefficient_history.npy",
        name="Coefficient history",
    )

    analysis = analyze_lms_convergence(
        error,
        coefficient_history,
    )

    np.save(
        results_directory
        / "learning_curve.npy",
        analysis.learning_curve,
    )

    np.save(
        results_directory
        / "coefficient_change.npy",
        analysis.coefficient_change,
    )

    np.save(
        results_directory
        / "coefficient_change_norm.npy",
        analysis.coefficient_change_norm,
    )

    np.save(
        results_directory
        / "coefficient_change_curve.npy",
        analysis.coefficient_change_curve,
    )

    summary = {
        "module": "M3.03",
        "initial_error_power": (
            analysis.initial_error_power
        ),
        "final_error_power": (
            analysis.final_error_power
        ),
        "error_power_ratio": (
            analysis.error_power_ratio
        ),
        "maximum_coefficient_change": (
            analysis.maximum_coefficient_change
        ),
        "final_coefficient_change": (
            analysis.final_coefficient_change
        ),
        "coefficient_change_ratio": (
            analysis.coefficient_change_ratio
        ),
        "error_decreased": (
            analysis.error_decreased
        ),
        "coefficients_stabilized": (
            analysis.coefficients_stabilized
        ),
    }

    with (
        results_directory
        / "convergence_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print(
        "M3.03 LMS convergence analysis completed."
    )

    print()

    print(
        f"Initial error power: "
        f"{analysis.initial_error_power:.10g}"
    )

    print(
        f"Final error power: "
        f"{analysis.final_error_power:.10g}"
    )

    print(
        f"Error power ratio: "
        f"{analysis.error_power_ratio:.10g}"
    )

    print()

    print(
        f"Error decreased: "
        f"{analysis.error_decreased}"
    )

    print(
        f"Coefficients stabilized: "
        f"{analysis.coefficients_stabilized}"
    )


if __name__ == "__main__":
    main()