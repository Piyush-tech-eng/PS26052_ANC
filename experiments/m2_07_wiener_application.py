from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.statistics import (
    apply_fir,
    load_module1_handoff,
    mean_squared_error,
    normalized_correlation,
    root_mean_squared_error,
)


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    # -----------------------------------------
    # Inputs
    # -----------------------------------------

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

    coefficient_path = (
        wiener_directory
        / "wiener_coefficients.npy"
    )

    # -----------------------------------------
    # Outputs
    # -----------------------------------------

    results_directory = (
        project_root
        / "results"
        / "m2_07_wiener_application"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------
    # Load frozen Module 1 handoff
    # -----------------------------------------

    handoff = load_module1_handoff(
        handoff_directory
    )

    x = handoff.reference
    d = handoff.desired

    # -----------------------------------------
    # Load M2.06 Wiener coefficients
    # -----------------------------------------

    w_opt = np.load(
        coefficient_path,
        allow_pickle=False,
    )

    w_opt = np.asarray(
        w_opt,
        dtype=np.float64,
    )

    if w_opt.ndim != 1:
        raise ValueError(
            "Wiener coefficient vector must be one-dimensional."
        )

    if not np.isfinite(w_opt).all():
        raise ValueError(
            "Wiener coefficient vector contains NaN or Inf."
        )

    # -----------------------------------------
    # Apply Wiener filter
    #
    # d_hat[n] = w_opt[n] * x[n]
    #             in the FIR convolution sense
    # -----------------------------------------

    d_hat = apply_fir(
        x,
        w_opt,
    )

    # -----------------------------------------
    # Residual
    #
    # e[n] = d[n] - d_hat[n]
    # -----------------------------------------

    error = (
        d - d_hat
    )

    # -----------------------------------------
    # Metrics
    # -----------------------------------------

    mse = mean_squared_error(
        d,
        d_hat,
    )

    rmse = root_mean_squared_error(
        d,
        d_hat,
    )

    correlation = normalized_correlation(
        d,
        d_hat,
    )

    target_power = float(
        np.mean(d ** 2)
    )

    residual_power = float(
        np.mean(error ** 2)
    )

    relative_mse = float(
        mse
        / (
            target_power
            + np.finfo(np.float64).eps
        )
    )

    explained_power_fraction = float(
        1.0 - relative_mse
    )

    # -----------------------------------------
    # Save numerical artifacts
    # -----------------------------------------

    np.save(
        results_directory
        / "estimated_desired.npy",
        d_hat,
    )

    np.save(
        results_directory
        / "error.npy",
        error,
    )

    # -----------------------------------------
    # Save summary
    # -----------------------------------------

    summary = {
        "module": "M2.07",
        "equation": (
            "d_hat[n] = sum_k w_opt[k] x[n-k]"
        ),
        "error_equation": (
            "e[n] = d[n] - d_hat[n]"
        ),
        "sampling_rate_hz": (
            handoff.sampling_rate_hz
        ),
        "num_samples": (
            handoff.num_samples
        ),
        "filter_length": int(
            len(w_opt)
        ),
        "metrics": {
            "mse": mse,
            "rmse": rmse,
            "normalized_correlation": correlation,
            "desired_power": target_power,
            "residual_power": residual_power,
            "relative_mse": relative_mse,
            "explained_power_fraction": (
                explained_power_fraction
            ),
        },
        "files": {
            "estimated_desired": (
                "estimated_desired.npy"
            ),
            "error": (
                "error.npy"
            ),
        },
    }

    summary_path = (
        results_directory
        / "wiener_application_summary.json"
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
        "M2.07 — Wiener Filter Application"
    )

    print(
        "----------------------------------"
    )

    print(
        f"Filter length: {len(w_opt)}"
    )

    print(
        f"MSE: {mse:.10g}"
    )

    print(
        f"RMSE: {rmse:.10g}"
    )

    print(
        "Normalized correlation: "
        f"{correlation:.10g}"
    )

    print(
        f"Desired power: "
        f"{target_power:.10g}"
    )

    print(
        f"Residual power: "
        f"{residual_power:.10g}"
    )

    print(
        f"Relative MSE: "
        f"{relative_mse:.10g}"
    )

    print(
        f"Explained power fraction: "
        f"{explained_power_fraction:.10g}"
    )

    print(
        f"Results saved to: "
        f"{results_directory}"
    )


if __name__ == "__main__":
    main()