"""M6.01 — Identified Secondary-Path ANC Integration Experiment.

This experiment is the first Module 6 integration checkpoint.

Pipeline:

    Independent identification excitation
                ↓
        true secondary path S(z)
                ↓
       measured path response
                ↓
    Module 5 identification
                ↓
        estimated path Ŝ(z)
                ↓
    SecondaryPathModel artifact
                ↓
    Module 4 ANC plant
                ↓
          FxNLMS ANC
                ↓
      residual error e[n]

The experiment compares:

    1. No control
    2. FxNLMS using perfect secondary-path knowledge
    3. FxNLMS using the secondary-path estimate from Module 5

Important:

The true secondary path is used only by the simulation to generate
the physical error signal.

The identified path is used by the adaptive ANC algorithm.

This is an offline integration experiment. It does not yet represent
live microphone streaming.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from anc.plant.digital import (
    ANCExperimentConfig,
    run_anc_experiment,
)
from anc.secondary_path.identification import (
    IdentificationConfig,
    SecondaryPathModel,
    identify_secondary_path,
    model_from_identification,
    validate_secondary_path_estimate,
)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------


def save_json(
    path: Path,
    payload: dict,
) -> None:
    """Save JSON with consistent formatting."""

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
        )


def power_db(
    signal: np.ndarray,
) -> float:
    """Return mean-square power in dB."""

    power = float(
        np.mean(signal**2)
    )

    return float(
        10.0
        * np.log10(
            power
            + np.finfo(
                np.float64
            ).eps
        )
    )


def noise_reduction_db(
    baseline: np.ndarray,
    residual: np.ndarray,
) -> float:
    """Compute residual noise reduction."""

    baseline_power = float(
        np.mean(baseline**2)
    )

    residual_power = float(
        np.mean(residual**2)
    )

    return float(
        10.0
        * np.log10(
            (
                baseline_power
                + np.finfo(
                    np.float64
                ).eps
            )
            /
            (
                residual_power
                + np.finfo(
                    np.float64
                ).eps
            )
        )
    )


def save_impulse_response_plot(
    true_path: np.ndarray,
    estimated_path: np.ndarray,
    output_path: Path,
) -> None:
    """Plot true and identified secondary paths."""

    samples = np.arange(
        len(true_path)
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.stem(
        samples,
        true_path,
        linefmt="C0-",
        markerfmt="C0o",
        basefmt=" ",
        label="True secondary path S(z)",
    )

    plt.stem(
        samples,
        estimated_path,
        linefmt="C1-",
        markerfmt="C1s",
        basefmt=" ",
        label="Identified path Ŝ(z)",
    )

    plt.title(
        "M6.01 — Secondary Path: "
        "True vs Identified"
    )

    plt.xlabel(
        "Sample"
    )

    plt.ylabel(
        "Impulse-response amplitude"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def save_error_power_plot(
    no_control_power: np.ndarray,
    perfect_model_power: np.ndarray,
    identified_model_power: np.ndarray,
    sampling_rate_hz: int,
    output_path: Path,
) -> None:
    """Plot error power histories."""

    time = (
        np.arange(
            len(no_control_power)
        )
        / sampling_rate_hz
    )

    plt.figure(
        figsize=(11, 6)
    )

    floor = (
        np.finfo(
            np.float64
        ).eps
    )

    plt.plot(
        time,
        10.0
        * np.log10(
            no_control_power
            + floor
        ),
        label="No control",
        linewidth=1.5,
    )

    plt.plot(
        time,
        10.0
        * np.log10(
            perfect_model_power
            + floor
        ),
        label="FxNLMS — perfect S(z)",
        linewidth=1.5,
    )

    plt.plot(
        time,
        10.0
        * np.log10(
            identified_model_power
            + floor
        ),
        label="FxNLMS — identified Ŝ(z)",
        linewidth=1.5,
    )

    plt.title(
        "M6.01 — ANC Error Power"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "Causal mean-square error power (dB)"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def save_error_waveform_plot(
    no_control: np.ndarray,
    perfect_model: np.ndarray,
    identified_model: np.ndarray,
    sampling_rate_hz: int,
    output_path: Path,
) -> None:
    """Plot a late-stage residual waveform comparison."""

    num_samples = len(
        no_control
    )

    display_length = min(
        num_samples,
        int(
            0.15
            * sampling_rate_hz
        ),
    )

    start = (
        num_samples
        - display_length
    )

    time = (
        np.arange(
            display_length
        )
        / sampling_rate_hz
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        time,
        no_control[start:],
        label="No control",
        alpha=0.8,
    )

    plt.plot(
        time,
        perfect_model[start:],
        label="Perfect S(z)",
        alpha=0.8,
    )

    plt.plot(
        time,
        identified_model[start:],
        label="Identified Ŝ(z)",
        alpha=0.8,
    )

    plt.title(
        "M6.01 — Late-Stage Error Waveforms"
    )

    plt.xlabel(
        "Time within displayed segment (seconds)"
    )

    plt.ylabel(
        "Error amplitude"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


def save_coefficient_norm_plot(
    perfect_history: np.ndarray,
    identified_history: np.ndarray,
    sampling_rate_hz: int,
    output_path: Path,
) -> None:
    """Plot controller coefficient-vector norm."""

    perfect_norm = np.linalg.norm(
        perfect_history,
        axis=1,
    )

    identified_norm = np.linalg.norm(
        identified_history,
        axis=1,
    )

    time = (
        np.arange(
            len(perfect_norm)
        )
        / sampling_rate_hz
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.plot(
        time,
        perfect_norm,
        label="Perfect S(z)",
    )

    plt.plot(
        time,
        identified_norm,
        label="Identified Ŝ(z)",
    )

    plt.title(
        "M6.01 — Controller Coefficient Norm"
    )

    plt.xlabel(
        "Time (seconds)"
    )

    plt.ylabel(
        "||w[n]||₂"
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
    )

    plt.close()


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------


def main() -> None:

    # -----------------------------------------------------
    # Project directories
    # -----------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    results_dir = (
        project_root
        / "results"
        / "m6_01_identified_secondary_path_anc"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Reproducibility
    # -----------------------------------------------------

    experiment_seed = 26052

    rng = np.random.default_rng(
        experiment_seed
    )

    # -----------------------------------------------------
    # Sampling configuration
    # -----------------------------------------------------

    sampling_rate_hz = 8_000

    duration_seconds = 8.0

    num_samples = int(
        sampling_rate_hz
        * duration_seconds
    )

    # -----------------------------------------------------
    # Generate ANC reference signal
    # -----------------------------------------------------
    #
    # This is the disturbance reference x[n].
    #
    # It is deliberately independent from the identification
    # excitation used below.
    # -----------------------------------------------------

    white_reference = rng.normal(
        loc=0.0,
        scale=0.20,
        size=num_samples,
    )

    # Introduce temporal correlation so the experiment is not
    # only an ideal white-noise example.

    reference = np.empty(
        num_samples,
        dtype=np.float64,
    )

    previous = 0.0

    alpha = 0.88

    for index, sample in enumerate(
        white_reference
    ):
        previous = (
            alpha
            * previous
            + (
                1.0
                - alpha
            )
            * sample
        )

        reference[index] = previous

    # -----------------------------------------------------
    # Define synthetic primary path P(z)
    # -----------------------------------------------------

    primary_path = np.array(
        [
            0.0,
            0.0,
            0.85,
            0.35,
            -0.18,
            0.08,
            -0.03,
        ],
        dtype=np.float64,
    )

    # -----------------------------------------------------
    # Define hidden true secondary path S(z)
    #
    # This represents the physical path in the simulation.
    #
    # It must NOT be supplied to the identification algorithm.
    # -----------------------------------------------------

    secondary_path_true = np.array(
        [
            0.0,
            0.0,
            0.55,
            0.25,
            -0.12,
            0.06,
            0.02,
        ],
        dtype=np.float64,
    )

    # -----------------------------------------------------
    # MODULE 5 HANDOFF
    #
    # Identify Ŝ(z) using independent excitation and
    # measured response.
    # -----------------------------------------------------

    identification_excitation = rng.normal(
        loc=0.0,
        scale=0.20,
        size=num_samples,
    )

    measured_response = np.convolve(
        identification_excitation,
        secondary_path_true,
        mode="full",
    )[
        :num_samples
    ]

    # Small measurement noise represents a measured response
    # rather than an exact numerical copy.

    measurement_noise = rng.normal(
        loc=0.0,
        scale=0.002,
        size=num_samples,
    )

    measured_response = (
        measured_response
        + measurement_noise
    )

    identification_config = (
        IdentificationConfig(
            filter_length=len(
                secondary_path_true
            ),
            step_size=0.5,
            algorithm="nlms",
            epsilon=1e-8,
            sampling_rate_hz=(
                sampling_rate_hz
            ),
            learning_window=256,
            experiment_seed=(
                experiment_seed
            ),
        )
    )

    identification_result = (
        identify_secondary_path(
            identification_excitation,
            measured_response,
            identification_config,
        )
    )

    validation = (
        validate_secondary_path_estimate(
            secondary_path_true,
            identification_result.secondary_path_estimate,
        )
    )

    identified_model = (
        model_from_identification(
            identification_result,
            validation,
            metadata={
                "module": "M6.01",
                "purpose": (
                    "Secondary-path handoff "
                    "for ANC integration"
                ),
                "experiment_seed": (
                    experiment_seed
                ),
            },
        )
    )

    model_path = (
        results_dir
        / "identified_secondary_path_model.npz"
    )

    identified_model.save(
        model_path
    )

    # Reload the saved artifact.
    #
    # The ANC experiment therefore consumes the portable
    # Module 5 handoff rather than the in-memory object.

    loaded_model = (
        SecondaryPathModel.load(
            model_path
        )
    )

    # -----------------------------------------------------
    # ANC controller configuration
    # -----------------------------------------------------

    no_control_config = (
        ANCExperimentConfig(
            filter_length=32,
            step_size=0.10,
            algorithm="none",
            sampling_rate_hz=(
                sampling_rate_hz
            ),
            learning_window=256,
        )
    )

    fxnlms_config = (
        ANCExperimentConfig(
            filter_length=32,
            step_size=0.20,
            algorithm="fxnlms",
            epsilon=1e-8,
            sampling_rate_hz=(
                sampling_rate_hz
            ),
            learning_window=256,
        )
    )

    # -----------------------------------------------------
    # CASE 1 — No control
    # -----------------------------------------------------

    no_control = (
        run_anc_experiment(
            reference,
            primary_path,
            secondary_path_true,
            loaded_model.impulse_response,
            no_control_config,
        )
    )

    # -----------------------------------------------------
    # CASE 2 — Perfect secondary-path model
    #
    # Ŝ(z) = S(z)
    #
    # Best-case simulation reference.
    # -----------------------------------------------------

    perfect_model = (
        run_anc_experiment(
            reference,
            primary_path,
            secondary_path_true,
            secondary_path_true,
            fxnlms_config,
        )
    )

    # -----------------------------------------------------
    # CASE 3 — Module 5 identified path
    #
    # Ŝ(z) ≈ S(z)
    # -----------------------------------------------------

    identified_path = (
        run_anc_experiment(
            reference,
            primary_path,
            secondary_path_true,
            loaded_model.impulse_response,
            fxnlms_config,
        )
    )

    # -----------------------------------------------------
    # Late-stage comparison
    # -----------------------------------------------------

    comparison_samples = min(
        4_000,
        num_samples,
    )

    baseline_window = (
        no_control.error[
            -comparison_samples:
        ]
    )

    perfect_window = (
        perfect_model.error[
            -comparison_samples:
        ]
    )

    identified_window = (
        identified_path.error[
            -comparison_samples:
        ]
    )

    perfect_nr_db = (
        noise_reduction_db(
            baseline_window,
            perfect_window,
        )
    )

    identified_nr_db = (
        noise_reduction_db(
            baseline_window,
            identified_window,
        )
    )

    # -----------------------------------------------------
    # Acceptance checks
    # -----------------------------------------------------

    if not np.isfinite(
        identified_path.error
    ).all():
        raise RuntimeError(
            "Identified-path ANC produced "
            "non-finite error samples."
        )

    if not np.isfinite(
        identified_path.final_coefficients
    ).all():
        raise RuntimeError(
            "Identified-path ANC produced "
            "non-finite coefficients."
        )

    if identified_nr_db <= 0.0:
        raise RuntimeError(
            "Acceptance failure: identified "
            "secondary-path ANC did not reduce "
            "late-stage residual power."
        )

    # -----------------------------------------------------
    # Save numerical artifacts
    # -----------------------------------------------------

    np.save(
        results_dir
        / "reference.npy",
        reference,
    )

    np.save(
        results_dir
        / "primary_path.npy",
        primary_path,
    )

    np.save(
        results_dir
        / "secondary_path_true.npy",
        secondary_path_true,
    )

    np.save(
        results_dir
        / "secondary_path_estimate.npy",
        loaded_model.impulse_response,
    )

    np.save(
        results_dir
        / "no_control_error.npy",
        no_control.error,
    )

    np.save(
        results_dir
        / "perfect_model_error.npy",
        perfect_model.error,
    )

    np.save(
        results_dir
        / "identified_model_error.npy",
        identified_path.error,
    )

    np.save(
        results_dir
        / "perfect_model_coefficients.npy",
        perfect_model.final_coefficients,
    )

    np.save(
        results_dir
        / "identified_model_coefficients.npy",
        identified_path.final_coefficients,
    )

    # -----------------------------------------------------
    # Save plots
    # -----------------------------------------------------

    save_impulse_response_plot(
        secondary_path_true,
        loaded_model.impulse_response,
        results_dir
        / "secondary_path_true_vs_estimate.png",
    )

    save_error_power_plot(
        no_control.error_power,
        perfect_model.error_power,
        identified_path.error_power,
        sampling_rate_hz,
        results_dir
        / "error_power_comparison.png",
    )

    save_error_waveform_plot(
        no_control.error,
        perfect_model.error,
        identified_path.error,
        sampling_rate_hz,
        results_dir
        / "late_stage_error_waveforms.png",
    )

    save_coefficient_norm_plot(
        perfect_model.coefficient_history,
        identified_path.coefficient_history,
        sampling_rate_hz,
        results_dir
        / "controller_coefficient_norm.png",
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    summary = {
        "module": "M6.01",
        "experiment": (
            "identified_secondary_path_anc"
        ),
        "experiment_seed": (
            experiment_seed
        ),
        "sampling_rate_hz": (
            sampling_rate_hz
        ),
        "duration_seconds": (
            duration_seconds
        ),
        "num_samples": (
            num_samples
        ),
        "controller": {
            "algorithm": "fxnlms",
            "filter_length": 32,
            "step_size": 0.20,
            "epsilon": 1e-8,
        },
        "identification": {
            "algorithm": (
                identification_config.algorithm
            ),
            "filter_length": (
                identification_config.filter_length
            ),
            "step_size": (
                identification_config.step_size
            ),
        },
        "secondary_path_validation": {
            "impulse_response_rmse": (
                validation.impulse_response_rmse
            ),
            "relative_impulse_error": (
                validation.relative_impulse_error
            ),
            "magnitude_response_rmse_db": (
                validation.magnitude_response_rmse_db
            ),
            "phase_response_rmse_radians": (
                validation.phase_response_rmse_radians
            ),
        },
        "late_stage_results": {
            "no_control_power_db": (
                power_db(
                    baseline_window
                )
            ),
            "perfect_model_power_db": (
                power_db(
                    perfect_window
                )
            ),
            "identified_model_power_db": (
                power_db(
                    identified_window
                )
            ),
            "perfect_model_noise_reduction_db": (
                perfect_nr_db
            ),
            "identified_model_noise_reduction_db": (
                identified_nr_db
            ),
        },
        "acceptance": {
            "identified_error_is_finite": True,
            "identified_coefficients_are_finite": (
                True
            ),
            "identified_model_reduces_residual": (
                bool(
                    identified_nr_db
                    > 0.0
                )
            ),
        },
    }

    save_json(
        results_dir
        / "summary.json",
        summary,
    )

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print()

    print(
        "M6.01 — Identified Secondary-Path "
        "ANC Integration"
    )

    print(
        "---------------------------------------"
    )

    print()

    print(
        "Module 5 identification"
    )

    print(
        f"Impulse-response RMSE: "
        f"{validation.impulse_response_rmse:.8f}"
    )

    print(
        f"Relative impulse error: "
        f"{validation.relative_impulse_error:.8f}"
    )

    print()

    print(
        "Late-stage ANC performance"
    )

    print(
        f"No-control power: "
        f"{power_db(baseline_window):.3f} dB"
    )

    print(
        f"Perfect-model reduction: "
        f"{perfect_nr_db:.3f} dB"
    )

    print(
        f"Identified-model reduction: "
        f"{identified_nr_db:.3f} dB"
    )

    print()

    print(
        "M6.01 acceptance passed."
    )

    print()

    print(
        f"Results saved to:\n"
        f"{results_dir}"
    )


if __name__ == "__main__":
    main()