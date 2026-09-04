from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.statistics import (
    autocorrelation,
    load_module1_handoff,
)


def main() -> None:
    project_root = (
        Path(__file__).resolve().parents[1]
    )

    handoff_directory = (
        project_root
        / "results"
        / "m1_handoff"
    )

    results_directory = (
        project_root
        / "results"
        / "m2_03_autocorrelation"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    handoff = load_module1_handoff(
        handoff_directory
    )

    x = handoff.reference

    max_lag = min(
        200,
        len(x) - 1,
    )

    lags, biased_values = autocorrelation(
        x,
        max_lag=max_lag,
        unbiased=False,
    )

    _, unbiased_values = autocorrelation(
        x,
        max_lag=max_lag,
        unbiased=True,
    )

    # Save numerical results.
    np.save(
        results_directory / "lags.npy",
        lags,
    )

    np.save(
        results_directory
        / "autocorrelation_biased.npy",
        biased_values,
    )

    np.save(
        results_directory
        / "autocorrelation_unbiased.npy",
        unbiased_values,
    )

    summary = {
        "module": "M2.03",
        "signal": "reference x[n]",
        "sampling_rate_hz": (
            handoff.sampling_rate_hz
        ),
        "num_samples": handoff.num_samples,
        "max_lag": int(max_lag),
        "estimator_default": "biased",
        "lag_zero": {
            "autocorrelation": float(
                biased_values[0]
            ),
            "mean_square": float(
                np.mean(x ** 2)
            ),
        },
        "files": {
            "lags": "lags.npy",
            "biased": "autocorrelation_biased.npy",
            "unbiased": "autocorrelation_unbiased.npy",
        },
    }

    with (
        results_directory
        / "autocorrelation_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )
        file.write("\n")

    print(
        "M2.03 — Autocorrelation"
    )
    print("------------------------")
    print(
        f"Signal: reference x[n]"
    )
    print(
        f"Maximum lag: {max_lag}"
    )
    print(
    f"Rxx[0]: {biased_values[0]:.10g}"
    )

    print(
        f"Mean-square: "
        f"{np.mean(x ** 2):.10g}"
    )

    print(
        "Rxx[0] equals mean-square: "
        f"{np.isclose(biased_values[0], np.mean(x ** 2))}"
    )

    print(
        f"Results saved to: "
        f"{results_directory}"
    )

if __name__ == "__main__":
    main()