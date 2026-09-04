from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from anc.statistics.correlation import (
    autocorrelation,
    cross_correlation,
)

from anc.statistics.basic import (
    load_module1_handoff,
)

from anc.statistics.wiener import (
    build_correlation_matrix,
    build_cross_correlation_vector,
)


def main() -> None:

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
        / "m2_05_wiener_structures"
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
    # Wiener filter configuration
    # -----------------------------------------

    filter_length = 32

    # -----------------------------------------
    # Estimate required correlation sequences
    # -----------------------------------------

    _, rxx = autocorrelation(
        x,
        max_lag=filter_length - 1,
        unbiased=False,
    )

    _, rxd = cross_correlation(
        x,
        d,
        max_lag=filter_length - 1,
        unbiased=False,
    )

    # -----------------------------------------
    # Construct Wiener system
    # -----------------------------------------

    correlation_matrix = (
        build_correlation_matrix(
            rxx,
            filter_length,
        )
    )

    cross_correlation_vector = (
        build_cross_correlation_vector(
            rxd,
            filter_length,
        )
    )

    # -----------------------------------------
    # Structural validation
    # -----------------------------------------

    matrix_is_symmetric = bool(
        np.allclose(
            correlation_matrix,
            correlation_matrix.T,
        )
    )

    matrix_shape_correct = (
        correlation_matrix.shape
        == (
            filter_length,
            filter_length,
        )
    )

    vector_shape_correct = (
        cross_correlation_vector.shape
        == (filter_length,)
    )

    # -----------------------------------------
    # Save numerical artifacts
    # -----------------------------------------

    np.save(
        results_directory
        / "correlation_matrix.npy",
        correlation_matrix,
    )

    np.save(
        results_directory
        / "cross_correlation_vector.npy",
        cross_correlation_vector,
    )

    # -----------------------------------------
    # Save summary
    # -----------------------------------------

    summary = {
        "module": "M2.05",
        "filter_length": filter_length,
        "sampling_rate_hz": (
            handoff.sampling_rate_hz
        ),
        "num_samples": (
            handoff.num_samples
        ),
        "matrix": {
            "shape": list(
                correlation_matrix.shape
            ),
            "symmetric": (
                matrix_is_symmetric
            ),
        },
        "vector": {
            "shape": list(
                cross_correlation_vector.shape
            ),
        },
        "equation": (
            "R @ w_opt = p"
        ),
        "files": {
            "correlation_matrix": (
                "correlation_matrix.npy"
            ),
            "cross_correlation_vector": (
                "cross_correlation_vector.npy"
            ),
        },
    }

    summary_path = (
        results_directory
        / "wiener_structures_summary.json"
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
        "M2.05 — Wiener Structures"
    )

    print("-------------------------")

    print(
        f"Filter length: {filter_length}"
    )

    print(
        f"R shape: "
        f"{correlation_matrix.shape}"
    )

    print(
        f"p shape: "
        f"{cross_correlation_vector.shape}"
    )

    print(
        f"R symmetric: "
        f"{matrix_is_symmetric}"
    )

    print(
        f"R @ w_opt = p"
    )

    print(
        f"Results saved to: "
        f"{results_directory}"
    )


if __name__ == "__main__":
    main()