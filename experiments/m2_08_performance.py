from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.statistics import (
    explained_power_fraction,
    load_module1_handoff,
    orthogonality_residual,
    relative_residual_power,
    signal_power,
)


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    # -----------------------------------------
    # M2.07 input
    # -----------------------------------------

    handoff_directory = (
        project_root
        / "results"
        / "m1_handoff"
    )

    application_directory = (
        project_root
        / "results"
        / "m2_07_wiener_application"
    )

    estimated_path = (
        application_directory
        / "estimated_desired.npy"
    )

    error_path = (
        application_directory
        / "error.npy"
    )

    # -----------------------------------------
    # M2.08 output
    # -----------------------------------------

    results_directory = (
        project_root
        / "results"
        / "m2_08_performance"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # Load data
    # -----------------------------------------

    handoff = load_module1_handoff(
        handoff_directory
    )

    d = handoff.desired

    d_hat = np.load(
        estimated_path,
        allow_pickle=False,
    )

    error = np.load(
        error_path,
        allow_pickle=False,
    )

    d_hat = np.asarray(
        d_hat,
        dtype=np.float64,
    )

    error = np.asarray(
        error,
        dtype=np.float64,
    )

    # -----------------------------------------
    # Validate loaded artifacts
    # -----------------------------------------

    if d_hat.shape != d.shape:
        raise ValueError(
            "Estimated desired signal has an unexpected shape."
        )

    if error.shape != d.shape:
        raise ValueError(
            "Error signal has an unexpected shape."
        )

    if not np.isfinite(d_hat).all():
        raise ValueError(
            "Estimated desired signal contains NaN or Inf."
        )

    if not np.isfinite(error).all():
        raise ValueError(
            "Error signal contains NaN or Inf."
        )

    # -----------------------------------------
    # Signal powers
    # -----------------------------------------

    desired_power = signal_power(
        d
    )

    estimate_power = signal_power(
        d_hat
    )

    error_power = signal_power(
        error
    )

    relative_error_power = (
        relative_residual_power(
            d,
            error,
        )
    )

    explained_fraction = (
        explained_power_fraction(
            d,
            error,
        )
    )

    # -----------------------------------------
    # Orthogonality principle
    #
    # E[e[n] x[n-k]] ≈ 0
    # -----------------------------------------

    filter_length = 32

    lags, orthogonality_values = (
        orthogonality_residual(
            error,
            handoff.reference,
            max_lag=filter_length - 1,
        )
    )

    max_absolute_orthogonality = float(
        np.max(
            np.abs(
                orthogonality_values
            )
        )
    )

    rms_orthogonality = float(
        np.sqrt(
            np.mean(
                orthogonality_values ** 2
            )
        )
    )

    # -----------------------------------------
    # Save correlation diagnostic
    # -----------------------------------------

    np.save(
        results_directory
        / "orthogonality_lags.npy",
        lags,
    )

    np.save(
        results_directory
        / "error_reference_correlation.npy",
        orthogonality_values,
    )

    # -----------------------------------------
    # Save summary
    # -----------------------------------------

    summary = {
        "module": "M2.08",
        "filter_length": filter_length,
        "sampling_rate_hz": (
            handoff.sampling_rate_hz
        ),
        "num_samples": (
            handoff.num_samples
        ),
        "power_analysis": {
            "desired_power": desired_power,
            "estimated_desired_power": (
                estimate_power
            ),
            "error_power": error_power,
            "relative_error_power": (
                relative_error_power
            ),
            "explained_power_fraction": (
                explained_fraction
            ),
        },
        "orthogonality_analysis": {
            "equation": (
                "E[e[n] x[n-k]] = 0"
            ),
            "maximum_absolute_value": (
                max_absolute_orthogonality
            ),
            "rms_value": (
                rms_orthogonality
            ),
            "lags_evaluated": [
                int(lag)
                for lag in lags
            ],
        },
        "input_files": {
            "desired": "m1_handoff/desired.npy",
            "estimated_desired": (
                "m2_07_wiener_application/"
                "estimated_desired.npy"
            ),
            "error": (
                "m2_07_wiener_application/"
                "error.npy"
            ),
        },
        "output_files": {
            "orthogonality_lags": (
                "orthogonality_lags.npy"
            ),
            "error_reference_correlation": (
                "error_reference_correlation.npy"
            ),
        },
    }

    summary_path = (
        results_directory
        / "performance_summary.json"
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
        "M2.08 — Wiener Performance Analysis"
    )

    print(
        "------------------------------------"
    )

    print(
        f"Desired power: "
        f"{desired_power:.10g}"
    )

    print(
        f"Estimated desired power: "
        f"{estimate_power:.10g}"
    )

    print(
        f"Residual error power: "
        f"{error_power:.10g}"
    )

    print(
        f"Relative error power: "
        f"{relative_error_power:.10g}"
    )

    print(
        f"Explained power fraction: "
        f"{explained_fraction:.10g}"
    )

    print(
        "Maximum |E[e x]| over evaluated lags: "
        f"{max_absolute_orthogonality:.10g}"
    )

    print(
        "RMS E[e x] over evaluated lags: "
        f"{rms_orthogonality:.10g}"
    )

    print(
        f"Results saved to: "
        f"{results_directory}"
    )


if __name__ == "__main__":
    main()