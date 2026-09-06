"""M6.05 — ANC replay using the Module 5 identified secondary-path model.

This experiment connects:

    Module 5 secondary-path identification
            ->
    identified secondary-path artifact S_hat(z)
            ->
    Module 6 replay engine
            ->
    FxNLMS ANC replay
            ->
    residual evaluation

The experiment intentionally does not hardcode a Module 5 results
directory. Instead, it automatically searches the project's results/
directory for the identified secondary-path artifact.

Preferred artifact order:

    1. secondary_path_model.npy
    2. secondary_path_estimate.npy

An explicit --identified-model path may also be supplied.
"""

from __future__ import annotations

import argparse
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


def validate_vector(
    values: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    """Validate and return a finite one-dimensional float vector."""

    vector = np.asarray(
        values,
        dtype=np.float64,
    )

    if vector.ndim != 1:
        raise ValueError(
            f"{name} must be one-dimensional. "
            f"Got shape {vector.shape}."
        )

    if len(vector) == 0:
        raise ValueError(
            f"{name} must not be empty."
        )

    if not np.isfinite(
        vector
    ).all():
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    return vector.copy()


def load_secondary_path(
    path: Path,
) -> np.ndarray:
    """Load and validate a secondary-path .npy artifact."""

    try:

        values = np.load(
            path,
            allow_pickle=False,
        )

    except Exception as error:

        raise RuntimeError(
            "Failed to load identified "
            "secondary-path artifact:\n"
            f"{path}\n\n"
            f"Reason: {error}"
        ) from error

    return validate_vector(
        values,
        name="identified secondary-path model",
    )


def find_identified_model(
    project_root: Path,
    requested_path: Path | None,
) -> Path:
    """Find the Module 5 identified secondary-path artifact.

    Search priority:

        explicit path
            ->
        explicit filename search
            ->
        secondary_path_model.npy
            ->
        secondary_path_estimate.npy
    """

    results_root = (
        project_root
        / "results"
    )

    if requested_path is not None:

        candidate = requested_path

        # Relative paths are interpreted from the
        # project root.
        if not candidate.is_absolute():

            candidate = (
                project_root
                / candidate
            )

        candidate = (
            candidate.resolve()
        )

        if candidate.is_file():

            return candidate

        # If the exact path fails, search recursively
        # using only the requested filename.
        filename = (
            requested_path.name
        )

        matches = sorted(
            results_root.rglob(
                filename
            )
        )

        matches = [
            path
            for path in matches
            if path.is_file()
        ]

        if len(matches) == 1:

            return (
                matches[0]
                .resolve()
            )

        if len(matches) > 1:

            # Prefer a path containing "m5".
            m5_matches = [
                path
                for path in matches
                if "m5" in str(
                    path
                ).lower()
            ]

            if len(m5_matches) == 1:

                return (
                    m5_matches[0]
                    .resolve()
                )

            return (
                matches[0]
                .resolve()
            )

        print()

        print(
            "WARNING: The explicit "
            "--identified-model path "
            "was not found."
        )

        print(
            f"Requested path: "
            f"{candidate}"
        )

        print(
            "Searching automatically "
            "inside results/..."
        )

    if not results_root.is_dir():

        raise FileNotFoundError(
            "The project's results directory "
            "does not exist:\n"
            f"{results_root}"
        )

    preferred_filenames = [
        "secondary_path_model.npy",
        "secondary_path_estimate.npy",
    ]

    for filename in (
        preferred_filenames
    ):

        matches = sorted(
            results_root.rglob(
                filename
            )
        )

        matches = [
            path
            for path in matches
            if path.is_file()
        ]

        if len(matches) == 0:

            continue

        # Prefer Module 5 result directories.
        m5_matches = [
            path
            for path in matches
            if "m5" in str(
                path
            ).lower()
        ]

        if len(m5_matches) > 0:

            return (
                m5_matches[0]
                .resolve()
            )

        return (
            matches[0]
            .resolve()
        )

    searched = []

    for filename in (
        preferred_filenames
    ):

        searched.append(
            filename
        )

    raise FileNotFoundError(
        "Could not automatically find a "
        "Module 5 identified secondary-path "
        "artifact.\n\n"
        "Searched recursively inside:\n"
        f"{results_root}\n\n"
        "Expected one of:\n"
        + "\n".join(
            f"  - {name}"
            for name in searched
        )
    )


def create_reference_signal(
    rng: np.random.Generator,
    num_samples: int,
) -> np.ndarray:
    """Create a reproducible correlated reference signal."""

    white = rng.normal(
        0.0,
        0.25,
        size=num_samples,
    )

    reference = np.empty(
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

        reference[index] = (
            previous
        )

    return reference


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "M6.05 ANC replay using the "
            "Module 5 identified secondary path."
        )
    )

    parser.add_argument(
        "--identified-model",
        type=Path,
        default=None,
        help=(
            "Optional path to the identified "
            "secondary-path .npy artifact. "
            "If omitted, the experiment searches "
            "results/ automatically."
        ),
    )

    args = parser.parse_args()

    # ---------------------------------------------
    # Project paths.
    # ---------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    results_dir = (
        project_root
        / "results"
        / "m6_05_identified_secondary_path_replay"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------
    # Find the ACTUAL Module 5 artifact.
    # ---------------------------------------------

    identified_model_path = (
        find_identified_model(
            project_root,
            args.identified_model,
        )
    )

    print()

    print(
        "M6.05 — Identified Secondary-Path Replay"
    )

    print(
        "----------------------------------------"
    )

    print()

    print(
        "Identified secondary-path artifact found:"
    )

    print(
        identified_model_path
    )

    # ---------------------------------------------
    # Load identified S_hat(z).
    # ---------------------------------------------

    secondary_path_model = (
        load_secondary_path(
            identified_model_path
        )
    )

    print()

    print(
        f"Identified model length: "
        f"{len(secondary_path_model)}"
    )

    # ---------------------------------------------
    # Experiment configuration.
    # ---------------------------------------------

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

    # ---------------------------------------------
    # Recorded-style reference.
    # ---------------------------------------------

    reference_signal = (
        create_reference_signal(
            rng,
            num_samples,
        )
    )

    # ---------------------------------------------
    # Simulated measured disturbance.
    #
    # This represents a recorded disturbance at the
    # error sensor.
    # ---------------------------------------------

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

    # ---------------------------------------------
    # True secondary path.
    #
    # This represents S(z), which is used only to
    # propagate the controller output physically.
    # ---------------------------------------------

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

    # ---------------------------------------------
    # Save the replay signals.
    # ---------------------------------------------

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

    # ---------------------------------------------
    # Load through M6.03 recording API.
    # ---------------------------------------------

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

    # ---------------------------------------------
    # ANC configuration.
    # ---------------------------------------------

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

    # ---------------------------------------------
    # Run M6.04 replay using:
    #
    # true S(z)
    # +
    # identified S_hat(z)
    # ---------------------------------------------

    result = run_anc_replay(
        replay_inputs,
        secondary_path_true,
        secondary_path_model,
        config,
    )

    # ---------------------------------------------
    # Acceptance checks.
    # ---------------------------------------------

    if not np.isfinite(
        result.residual
    ).all():

        raise RuntimeError(
            "Replay residual contains "
            "NaN or Inf."
        )

    if not np.isfinite(
        result.final_coefficients
    ).all():

        raise RuntimeError(
            "Final controller coefficients "
            "contain NaN or Inf."
        )

    if (
        result.final_residual_power
        >= result.initial_residual_power
    ):

        raise RuntimeError(
            "ANC replay did not reduce "
            "residual power.\n\n"
            "The Module 5 identified model "
            "was successfully loaded, but this "
            "specific model/configuration did "
            "not achieve convergence."
        )

    # ---------------------------------------------
    # Save artifacts.
    # ---------------------------------------------

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

    np.save(
        results_dir
        / "identified_secondary_path_used.npy",
        secondary_path_model,
    )

    # ---------------------------------------------
    # Plot signal comparison.
    # ---------------------------------------------

    display_samples = min(
        4_000,
        len(result.residual),
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
        label=(
            "Measured disturbance"
        ),
    )

    plt.plot(
        time,
        result.residual[
            :display_samples
        ],
        label=(
            "ANC residual"
        ),
        alpha=0.8,
    )

    plt.title(
        "M6.05 — ANC Replay Using Module 5 Identified Secondary Path"
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
        / "identified_model_replay.png",
        dpi=150,
    )

    plt.close()

    # ---------------------------------------------
    # Plot residual power.
    # ---------------------------------------------

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
        label=(
            "Residual power"
        ),
    )

    plt.title(
        "M6.05 — Residual Power Using Identified Secondary Path"
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

    # ---------------------------------------------
    # Save summary.
    # ---------------------------------------------

    summary = {
        "module": "M6.05",
        "experiment": (
            "identified_secondary_path_replay"
        ),
        "seed": seed,
        "sampling_rate_hz": (
            sampling_rate_hz
        ),
        "duration_seconds": (
            duration_seconds
        ),
        "num_samples": (
            len(result.reference)
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
        "identified_model_source": str(
            identified_model_path
        ),
        "identified_model_length": int(
            len(
                secondary_path_model
            )
        ),
        "true_secondary_path_length": int(
            len(
                secondary_path_true
            )
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
            "identified_model_loaded": True,
            "finite_residual": True,
            "finite_coefficients": True,
            "residual_power_reduced": True,
        },
    }

    summary_path = (
        results_dir
        / "summary.json"
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

        file.write(
            "\n"
        )

    # ---------------------------------------------
    # Final output.
    # ---------------------------------------------

    print()

    print(
        "Initial residual power:"
    )

    print(
        f"{result.initial_residual_power:.8e}"
    )

    print()

    print(
        "Final residual power:"
    )

    print(
        f"{result.final_residual_power:.8e}"
    )

    print()

    print(
        "Residual power ratio:"
    )

    print(
        f"{result.residual_power_ratio:.8f}"
    )

    print()

    print(
        "Attenuation:"
    )

    print(
        f"{result.attenuation_db:.3f} dB"
    )

    print()

    print(
        "M6.05 acceptance passed."
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        results_dir
    )


if __name__ == "__main__":
    main()