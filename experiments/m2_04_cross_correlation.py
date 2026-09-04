from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.statistics import (
    cross_correlation,
    load_module1_handoff,
)


def main() -> None:
    """Compute reference-desired cross-correlation."""

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    handoff_directory = (
        project_root
        / "results"
        / "m1_handoff"
    )

    results_directory = (
        project_root
        / "results"
        / "m2_04_cross_correlation"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    handoff = load_module1_handoff(
        handoff_directory
    )

    x = handoff.reference
    d = handoff.desired

    max_lag = min(
        200,
        len(x) - 1,
    )

    lags, biased_values = (
        cross_correlation(
            x,
            d,
            max_lag=max_lag,
            unbiased=False,
        )
    )

    _, unbiased_values = (
        cross_correlation(
            x,
            d,
            max_lag=max_lag,
            unbiased=True,
        )
    )

    # ---------------------------------------------
    # Save numerical artifacts
    # ---------------------------------------------

    np.save(
        results_directory / "lags.npy",
        lags,
    )

    np.save(
        results_directory
        / "cross_correlation_biased.npy",
        biased_values,
    )

    np.save(
        results_directory
        / "cross_correlation_unbiased.npy",
        unbiased_values,
    )

    # ---------------------------------------------
    # Validation
    #
    # At zero lag:
    #
    # Rxd[0] = mean(d[n] * x[n])
    # ---------------------------------------------

    zero_lag_mean_product = float(
        np.mean(d * x)
    )

    zero_lag_matches = bool(
        np.isclose(
            biased_values[0],
            zero_lag_mean_product,
        )
    )

    # ---------------------------------------------
    # Save experiment summary
    # ---------------------------------------------

    summary = {
        "module": "M2.04",
        "reference_signal": "x[n]",
        "desired_signal": "d[n]",
        "sampling_rate_hz": (
            handoff.sampling_rate_hz
        ),
        "num_samples": (
            handoff.num_samples
        ),
        "max_lag": int(max_lag),
        "estimator_default": "biased",
        "convention": (
            "Rxd[k] = (1/N) "
            "* sum(d[n] * x[n-k])"
        ),
        "lag_zero": {
            "cross_correlation": float(
                biased_values[0]
            ),
            "mean_product": (
                zero_lag_mean_product
            ),
            "matches": (
                zero_lag_matches
            ),
        },
        "files": {
            "lags": "lags.npy",
            "biased": (
                "cross_correlation_biased.npy"
            ),
            "unbiased": (
                "cross_correlation_unbiased.npy"
            ),
        },
    }

    summary_path = (
        results_directory
        / "cross_correlation_summary.json"
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

    # ---------------------------------------------
    # Report
    # ---------------------------------------------

    print(
        "M2.04 — Cross-Correlation"
    )

    print("--------------------------")

    print(
        "Reference signal: x[n]"
    )

    print(
        "Desired signal: d[n]"
    )

    print(
        f"Maximum lag: {max_lag}"
    )

    print(
        f"Rxd[0]: "
        f"{biased_values[0]:.10g}"
    )

    print(
        f"Mean(d[n] * x[n]): "
        f"{zero_lag_mean_product:.10g}"
    )

    print(
        "Rxd[0] equals mean product: "
        f"{zero_lag_matches}"
    )

    print(
        f"Results saved to: "
        f"{results_directory}"
    )


if __name__ == "__main__":
    main()