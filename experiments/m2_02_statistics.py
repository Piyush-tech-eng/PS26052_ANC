from __future__ import annotations

import json
from pathlib import Path

from anc.statistics.basic import (
    load_module1_handoff,
    summarize_signal_statistics,
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
        / "m2_02_statistics"
    )

    results_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    handoff = load_module1_handoff(
        handoff_directory
    )

    reference_stats = summarize_signal_statistics(
        handoff.reference
    )

    desired_stats = summarize_signal_statistics(
        handoff.desired
    )

    summary = {
        "module": "M2.02",
        "source_dataset": (
            handoff.metadata.get("dataset")
        ),
        "sampling_rate_hz": (
            handoff.sampling_rate_hz
        ),
        "num_samples": (
            handoff.num_samples
        ),
        "duration_seconds": (
            handoff.duration_seconds
        ),
        "reference_x_n": reference_stats,
        "desired_d_n": desired_stats,
        "relationships": {
            "reference_power_equals_variance_plus_mean_squared": (
                reference_stats["average_power"]
                == reference_stats["variance"]
                + reference_stats["mean"] ** 2
            ),
            "desired_power_equals_variance_plus_mean_squared": (
                desired_stats["average_power"]
                == desired_stats["variance"]
                + desired_stats["mean"] ** 2
            ),
            "reference_rms_squared_equals_power": (
                reference_stats["rms"] ** 2
                == reference_stats["average_power"]
            ),
            "desired_rms_squared_equals_power": (
                desired_stats["rms"] ** 2
                == desired_stats["average_power"]
            ),
        },
    }

    output_path = (
        results_directory
        / "statistics_summary.json"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=2,
        )
        file.write("\n")

    print("M2.02 — Signal Statistics")
    print("--------------------------")

    print("\nReference x[n]")
    for key, value in reference_stats.items():
        print(f"{key}: {value:.10g}")

    print("\nDesired d[n]")
    for key, value in desired_stats.items():
        print(f"{key}: {value:.10g}")

    print(
        f"\nStatistics saved to: {output_path}"
    )


if __name__ == "__main__":
    main()