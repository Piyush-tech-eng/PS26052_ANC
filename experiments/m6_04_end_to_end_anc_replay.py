"""M6.04 — End-to-End ANC Replay.

This experiment connects the recorded-input pipeline from Modules 6.02
and 6.03 to the existing ANC controller and secondary-path model.

Pipeline:

    recorded-style reference
            +
    recorded-style measured disturbance
                |
                v
        M6.03 replay preparation
                |
                v
        M6.04 ANC replay engine
                |
                +--> controller output
                |
                +--> secondary-path output
                |
                +--> residual
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anc.io import (
    load_npy_recording,
    prepare_replay_inputs,
)

from anc.plant.digital import (
    ANCExperimentConfig,
    apply_causal_fir,
)

from anc.replay import (
    run_anc_replay,
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
        / "m6_04_end_to_end_anc_replay"
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

    duration_seconds = 8.0

    num_samples = int(
        sampling_rate_hz
        * duration_seconds
    )

    # -------------------------------------------------
    # Recorded-style reference signal.
    # -------------------------------------------------

    white = rng.normal(
        0.0,
        0.25,
        size=num_samples,
    )

    reference_signal = np.empty(
        num_samples,
        dtype=np.float64,
    )

    previous = 0.0

    for index, sample in enumerate(
        white
    ):

        previous = (
            0.88
            * previous
            + 0.12
            * sample
        )

        reference_signal[index] = (
            previous
        )

    # -------------------------------------------------
    # Measured disturbance recording.
    #
    # For this validation experiment we create it through
    # a primary path, then save/load it as an independent
    # recording.
    # -------------------------------------------------

    primary_path = np.array(
        [
            0.0,
            0.0,
            0.75,
            0.30,
            -0.12,
            0.05,
        ],
        dtype=np.float64,
    )

    measured_signal = (
        apply_causal_fir(
            reference_signal,
            primary_path,
        )
    )

    measured_signal += (
        rng.normal(
            0.0,
            0.001,
            size=num_samples,
        )
    )

    # -------------------------------------------------
    # Save independent recording artifacts.
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

    replay_inputs = (
        prepare_replay_inputs(
            reference_recording,
            measured_recording,
            align=False,
        )
    )

    # -------------------------------------------------
    # Secondary path.
    # -------------------------------------------------

    secondary_path_true = np.array(
        [
            0.0,
            0.0,
            0.62,
            0.28,
            -0.10,
            0.04,
        ],
        dtype=np.float64,
    )

    # For M6.04 validation, use a close but not identical
    # model. Future modules can replace this with the
    # identified model artifact.
    secondary_path_model = np.array(
        [
            0.0,
            0.0,
            0.60,
            0.26,
            -0.09,
            0.04,
        ],
        dtype=np.float64,
    )

    # -------------------------------------------------
    # Existing ANC configuration.
    # -------------------------------------------------

    config = ANCExperimentConfig(
        filter_length=32,
        step_size=0.05,
        algorithm="fxnlms",
        epsilon=1e-8,
        sampling_rate_hz=(
            sampling_rate_hz
        ),
        learning_window=512,
    )

    # -------------------------------------------------
    # Run ANC replay.
    # -------------------------------------------------

    result = run_anc_replay(
        replay_inputs,
        secondary_path_true,
        secondary_path_model,
        config,
    )

    # -------------------------------------------------
    # Acceptance checks.
    # -------------------------------------------------

    if not np.isfinite(
        result.residual
    ).all():
        raise RuntimeError(
            "Replay residual contains "
            "non-finite values."
        )

    if not np.isfinite(
        result.final_coefficients
    ).all():
        raise RuntimeError(
            "Final controller coefficients "
            "contain non-finite values."
        )

    if (
        result.final_residual_power
        >= result.initial_residual_power
    ):
        raise RuntimeError(
            "ANC replay did not reduce "
            "residual power."
        )

    # -------------------------------------------------
    # Save signal artifacts.
    # -------------------------------------------------

    np.save(
        results_dir
        / "reference.npy",
        result.reference,
    )

    np.save(
        results_dir
        / "measured.npy",
        result.measured_disturbance,
    )

    np.save(
        results_dir
        / "controller_output.npy",
        result.controller_output,
    )

    np.save(
        results_dir
        / "secondary_output.npy",
        result.secondary_path_output,
    )

    np.save(
        results_dir
        / "residual.npy",
        result.residual,
    )

    np.save(
        results_dir
        / "filtered_reference.npy",
        result.filtered_reference,
    )

    np.save(
        results_dir
        / "final_coefficients.npy",
        result.final_coefficients,
    )

    # -------------------------------------------------
    # Plot replay signals.
    # -------------------------------------------------

    display_samples = min(
        3_000,
        len(
            result.reference
        ),
    )

    time = (
        np.arange(
            display_samples
        )
        / sampling_rate_hz
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        result.measured_disturbance[
            :display_samples
        ],
        label="Measured disturbance",
    )

    plt.plot(
        time,
        result.residual[
            :display_samples
        ],
        label="ANC residual",
        alpha=0.8,
    )

    plt.title(
        "M6.04 — Measured Disturbance vs ANC Residual"
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
        / "disturbance_vs_residual.png",
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------
    # Plot residual power.
    # -------------------------------------------------

    power_time = (
        np.arange(
            len(
                result.residual_power
            )
        )
        / sampling_rate_hz
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        power_time,
        result.residual_power,
        label="Residual power",
    )

    plt.title(
        "M6.04 — ANC Replay Residual Power"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Mean-square residual"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        results_dir
        / "residual_power.png",
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------
    # Summary.
    # -------------------------------------------------

    summary = {
        "module": "M6.04",
        "experiment": (
            "end_to_end_anc_replay"
        ),
        "seed": seed,
        "sampling_rate_hz": (
            sampling_rate_hz
        ),
        "duration_seconds": (
            duration_seconds
        ),
        "num_samples": (
            len(
                result.reference
            )
        ),
        "algorithm": (
            config.algorithm
        ),
        "filter_length": (
            config.filter_length
        ),
        "step_size": (
            config.step_size
        ),
        "alignment_applied": (
            result.alignment_applied
        ),
        "delay_samples": (
            result.delay_samples
        ),
        "initial_residual_power": (
            result.initial_residual_power
        ),
        "final_residual_power": (
            result.final_residual_power
        ),
        "residual_power_ratio": (
            result.residual_power_ratio
        ),
        "attenuation_db": (
            result.attenuation_db
        ),
        "acceptance": {
            "finite_residual": True,
            "finite_final_coefficients": True,
            "residual_power_reduced": True,
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
        "M6.04 — End-to-End ANC Replay"
    )

    print(
        "----------------------------------------"
    )

    print()

    print(
        f"Algorithm: "
        f"{config.algorithm}"
    )

    print(
        f"Samples: "
        f"{len(result.reference)}"
    )

    print()

    print(
        f"Initial residual power: "
        f"{result.initial_residual_power:.8e}"
    )

    print(
        f"Final residual power: "
        f"{result.final_residual_power:.8e}"
    )

    print()

    print(
        f"Residual power ratio: "
        f"{result.residual_power_ratio:.6f}"
    )

    print(
        f"Attenuation: "
        f"{result.attenuation_db:.3f} dB"
    )

    print()

    print(
        "M6.04 acceptance passed."
    )

    print()

    print(
        f"Results saved to:\n"
        f"{results_dir}"
    )


if __name__ == "__main__":
    main()