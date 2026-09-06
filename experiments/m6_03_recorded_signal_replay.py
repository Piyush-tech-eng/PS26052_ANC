"""M6.03 — Recorded Signal Input and ANC Replay Preparation."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anc.calibration import apply_known_delay
from anc.io import (
    load_npy_recording,
    prepare_replay_inputs,
)


def main() -> None:

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    results_dir = (
        project_root
        / "results"
        / "m6_03_recorded_signal_replay"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed = 26052

    rng = np.random.default_rng(
        seed
    )

    sampling_rate_hz = 8_000

    duration_seconds = 4.0

    num_samples = int(
        sampling_rate_hz
        * duration_seconds
    )

    # -------------------------------------------------
    # Simulate independently obtained recorded signals.
    # -------------------------------------------------

    white = rng.normal(
        0.0,
        0.20,
        size=num_samples,
    )

    reference_signal = np.empty(
        num_samples,
        dtype=np.float64,
    )

    previous = 0.0

    for index, value in enumerate(
        white
    ):
        previous = (
            0.90
            * previous
            + 0.10
            * value
        )

        reference_signal[index] = (
            previous
        )

    known_delay = 95

    measured_signal = (
        0.80
        * apply_known_delay(
            reference_signal,
            delay_samples=known_delay,
        )
    )

    measured_signal += (
        rng.normal(
            0.0,
            0.002,
            size=num_samples,
        )
    )

    # -------------------------------------------------
    # Save as independent recording artifacts.
    # -------------------------------------------------

    reference_path = (
        results_dir
        / "recorded_reference.npy"
    )

    measured_path = (
        results_dir
        / "recorded_measured.npy"
    )

    np.save(
        reference_path,
        reference_signal,
    )

    np.save(
        measured_path,
        measured_signal,
    )

    # -------------------------------------------------
    # Load through M6.03 interface.
    # -------------------------------------------------

    reference_recording = (
        load_npy_recording(
            reference_path,
            sampling_rate_hz=(
                sampling_rate_hz
            ),
            role="reference",
        )
    )

    measured_recording = (
        load_npy_recording(
            measured_path,
            sampling_rate_hz=(
                sampling_rate_hz
            ),
            role="measured",
        )
    )

    # -------------------------------------------------
    # Prepare replay inputs.
    # -------------------------------------------------

    replay = prepare_replay_inputs(
        reference_recording,
        measured_recording,
        align=True,
        max_delay_samples=500,
        minimum_overlap_samples=1_000,
    )

    # -------------------------------------------------
    # Acceptance.
    # -------------------------------------------------

    if (
        replay.delay_samples
        != known_delay
    ):
        raise RuntimeError(
            "Recorded replay preparation failed "
            "to recover the known timing offset."
        )

    if (
        replay.reference.num_samples
        != replay.measured.num_samples
    ):
        raise RuntimeError(
            "Replay inputs are not equal length."
        )

    if not np.isfinite(
        replay.reference.samples
    ).all():
        raise RuntimeError(
            "Reference replay signal contains "
            "non-finite values."
        )

    if not np.isfinite(
        replay.measured.samples
    ).all():
        raise RuntimeError(
            "Measured replay signal contains "
            "non-finite values."
        )

    # -------------------------------------------------
    # Save standardized replay arrays.
    # -------------------------------------------------

    np.save(
        results_dir
        / "replay_reference.npy",
        replay.reference.samples,
    )

    np.save(
        results_dir
        / "replay_measured.npy",
        replay.measured.samples,
    )

    # -------------------------------------------------
    # Plot aligned replay signals.
    # -------------------------------------------------

    display_samples = min(
        1_500,
        replay.overlap_samples,
    )

    time = (
        np.arange(
            display_samples
        )
        / sampling_rate_hz
    )

    plt.figure(
        figsize=(11, 5)
    )

    plt.plot(
        time,
        replay.reference.samples[
            :display_samples
        ],
        label="Replay reference",
    )

    plt.plot(
        time,
        replay.measured.samples[
            :display_samples
        ],
        label="Replay measured",
        alpha=0.75,
    )

    plt.title(
        "M6.03 — Standardized Recorded Replay Inputs"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "aligned_replay_inputs.png",
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------
    # Summary.
    # -------------------------------------------------

    summary = {
        "module": "M6.03",
        "experiment": (
            "recorded_signal_replay_preparation"
        ),
        "seed": seed,
        "sampling_rate_hz": (
            sampling_rate_hz
        ),
        "duration_seconds": (
            duration_seconds
        ),
        "known_delay_samples": (
            known_delay
        ),
        "estimated_delay_samples": (
            replay.delay_samples
        ),
        "alignment_applied": (
            replay.alignment_applied
        ),
        "overlap_samples": (
            replay.overlap_samples
        ),
        "reference": {
            "source": (
                replay.reference.source
            ),
            "samples": (
                replay.reference.num_samples
            ),
            "duration_seconds": (
                replay.reference.duration_seconds
            ),
        },
        "measured": {
            "source": (
                replay.measured.source
            ),
            "samples": (
                replay.measured.num_samples
            ),
            "duration_seconds": (
                replay.measured.duration_seconds
            ),
        },
        "acceptance": {
            "delay_recovered": True,
            "equal_length": True,
            "finite_reference": True,
            "finite_measured": True,
        },
    }

    with (
        results_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=2,
        )

        file.write(
            "\n"
        )

    print()

    print(
        "M6.03 — Recorded Signal Input "
        "and ANC Replay Preparation"
    )

    print(
        "----------------------------------------"
    )

    print()

    print(
        f"Known delay: "
        f"{known_delay} samples"
    )

    print(
        f"Recovered delay: "
        f"{replay.delay_samples} samples"
    )

    print(
        f"Replay overlap: "
        f"{replay.overlap_samples} samples"
    )

    print()

    print(
        "M6.03 acceptance passed."
    )

    print()

    print(
        f"Results saved to:\n"
        f"{results_dir}"
    )


if __name__ == "__main__":
    main()