from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from anc.adaptive import (
    run_lms_step_size_sweep,
)


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
        / "m3_04_lms_step_size_sweep"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference = np.load(
        handoff_directory
        / "reference.npy",
        allow_pickle=False,
    )

    desired = np.load(
        handoff_directory
        / "desired.npy",
        allow_pickle=False,
    )

    filter_length = 16

    step_sizes = [
        1e-5,
        1e-4,
        1e-3,
        1e-2,
    ]

    results = run_lms_step_size_sweep(
        reference,
        desired,
        filter_length=filter_length,
        step_sizes=step_sizes,
    )

    summary = {
        "module": "M3.04",
        "filter_length": filter_length,
        "runs": [
            asdict(result)
            for result in results
        ],
    }

    stable_runs = [
        result
        for result in results
        if result.stable
        and result.mse is not None
    ]

    if stable_runs:

        best_run = min(
            stable_runs,
            key=lambda result: (
                result.mse
                if result.mse is not None
                else np.inf
            ),
        )

        summary[
            "recommended_step_size"
        ] = best_run.step_size

    else:

        summary[
            "recommended_step_size"
        ] = None

    with (
        results_directory
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    for result in results:

        print()

        print(
            f"mu = {result.step_size}"
        )

        print(
            f"Stable: {result.stable}"
        )

        print(
            f"Converged: {result.converged}"
        )

        if result.stable:

            print(
                f"MSE: {result.mse:.10g}"
            )

            print(
                f"Error power ratio: "
                f"{result.error_power_ratio:.10g}"
            )

        else:

            print(
                f"Reason: {result.reason}"
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