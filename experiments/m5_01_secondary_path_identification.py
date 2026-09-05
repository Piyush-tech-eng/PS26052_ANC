"""Module 5 secondary-path identification and mismatch acceptance experiment."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anc.plant import ANCExperimentConfig, apply_causal_fir, make_delayed_fir_path, run_anc_experiment
from anc.secondary_path import (
    IdentificationConfig,
    SecondaryPathModel,
    identify_secondary_path,
    model_from_identification,
    perturb_secondary_path_model,
    validate_secondary_path_estimate,
)
from anc.visualization import save_line_plot


def validation_dict(validation) -> dict[str, object]:
    values = asdict(validation)
    values["impulse_response_error"] = validation.impulse_response_error.tolist()
    return values


def anc_metrics(run, baseline) -> dict[str, float]:
    return {
        "final_error_power": run.final_error_power,
        "final_power_vs_no_control": float(
            run.final_error_power
            / (baseline.final_error_power + np.finfo(np.float64).eps)
        ),
        "attenuation_db_vs_no_control": float(
            10.0
            * np.log10(
                (baseline.final_error_power + np.finfo(np.float64).eps)
                / (run.final_error_power + np.finfo(np.float64).eps)
            )
        ),
    }


def main() -> None:
    results_directory = PROJECT_ROOT / "results" / "m5_01_secondary_path_identification"
    results_directory.mkdir(parents=True, exist_ok=True)

    sampling_rate_hz = 8_000
    rng = np.random.default_rng(26052)

    # The identifier receives only probe and measured response.  Ground truth
    # is deliberately retained below for post-estimation validation only.
    secondary_path_true = make_delayed_fir_path(
        [0.55, 0.18, -0.08, 0.04],
        delay_samples=2,
    )
    excitation = rng.normal(0.0, 1.0, 18_000)
    measured_response = apply_causal_fir(excitation, secondary_path_true)
    identification_config = IdentificationConfig(
        filter_length=16,
        step_size=0.35,
        algorithm="nlms",
        epsilon=1e-8,
        sampling_rate_hz=sampling_rate_hz,
        learning_window=256,
        experiment_seed=26052,
    )
    identification = identify_secondary_path(
        excitation,
        measured_response,
        identification_config,
    )
    validation = validate_secondary_path_estimate(
        secondary_path_true,
        identification.secondary_path_estimate,
    )
    model = model_from_identification(
        identification,
        validation,
        metadata={
            "source": "synthetic-secondary-path-identification",
            "true_path_used_only_for": "post-identification validation",
        },
    )
    model_path = model.save(results_directory / "secondary_path_model.npz")
    reloaded_model = SecondaryPathModel.load(model_path)

    # Required numeric outputs and a human-readable metadata companion.
    np.save(results_directory / "identification_excitation.npy", identification.excitation)
    np.save(results_directory / "measured_secondary_response.npy", identification.measured_response)
    np.save(results_directory / "secondary_path_estimate.npy", identification.secondary_path_estimate)
    np.save(results_directory / "secondary_path_true.npy", secondary_path_true)
    np.save(results_directory / "secondary_path_error.npy", validation.impulse_response_error)
    np.save(results_directory / "identification_response_error.npy", identification.response_error)
    np.save(results_directory / "identification_mse_learning_curve.npy", identification.mse_learning_curve)
    np.save(results_directory / "identification_coefficient_history.npy", identification.coefficient_history)
    with (results_directory / "secondary_path_model_metadata.json").open("w", encoding="utf-8") as output:
        json.dump(model.to_metadata(), output, indent=2)

    # Visual validation of the identified impulse response and response errors.
    comparison_length = max(len(secondary_path_true), len(identification.secondary_path_estimate))
    padded_true = np.pad(secondary_path_true, (0, comparison_length - len(secondary_path_true)))
    padded_estimate = np.pad(
        identification.secondary_path_estimate,
        (0, comparison_length - len(identification.secondary_path_estimate)),
    )
    save_line_plot(
        results_directory / "secondary_path_impulse_comparison.png",
        {"True S(z)": padded_true, "Estimated S_hat(z)": padded_estimate},
        title="Secondary-path impulse response: true versus identified",
        xlabel="Tap index",
        ylabel="Coefficient",
    )
    frequency_bins = 2_048
    frequencies = np.fft.rfftfreq(frequency_bins, d=1.0 / sampling_rate_hz)
    true_frequency_response = np.fft.rfft(padded_true, n=frequency_bins)
    estimated_frequency_response = np.fft.rfft(padded_estimate, n=frequency_bins)
    save_line_plot(
        results_directory / "secondary_path_frequency_response.png",
        {"True S(z)": np.abs(true_frequency_response), "Estimated S_hat(z)": np.abs(estimated_frequency_response)},
        title="Secondary-path magnitude response",
        xlabel="Frequency (Hz)",
        ylabel="Magnitude",
        x_values=frequencies,
        logarithmic_y=True,
    )
    save_line_plot(
        results_directory / "secondary_path_phase_response.png",
        {"True S(z)": np.unwrap(np.angle(true_frequency_response)), "Estimated S_hat(z)": np.unwrap(np.angle(estimated_frequency_response))},
        title="Secondary-path phase response",
        xlabel="Frequency (Hz)",
        ylabel="Phase (radians)",
        x_values=frequencies,
    )
    save_line_plot(
        results_directory / "identification_convergence.png",
        {"Identification MSE": identification.mse_learning_curve},
        title="NLMS secondary-path identification convergence",
        ylabel="Causal MSE",
        logarithmic_y=True,
    )
    save_line_plot(
        results_directory / "secondary_path_response_error.png",
        {"Measured - estimated response": identification.response_error},
        title="Secondary-path identification residual",
        ylabel="Response error",
    )

    # Feed the validated model into the Module 4 plant, then deliberately
    # perturb it to expose the engineering cost of modelling mismatch.
    anc_reference = rng.normal(0.0, 1.0, 16_000)
    primary_path = make_delayed_fir_path([0.85, 0.25, -0.10], delay_samples=2)
    anc_config = ANCExperimentConfig(
        filter_length=32,
        step_size=0.25,
        algorithm="fxnlms",
        sampling_rate_hz=sampling_rate_hz,
    )
    no_control = run_anc_experiment(
        anc_reference,
        primary_path,
        secondary_path_true,
        reloaded_model.impulse_response,
        ANCExperimentConfig(32, 0.25, "none", sampling_rate_hz=sampling_rate_hz),
    )
    perturbation = np.zeros_like(reloaded_model.impulse_response)
    perturbation[2] = 0.18
    perturbation[3] = -0.10
    models = {
        "identified_perfect_case": reloaded_model.impulse_response,
        "gain_mismatch": perturb_secondary_path_model(reloaded_model.impulse_response, gain=0.65),
        "delay_mismatch": perturb_secondary_path_model(reloaded_model.impulse_response, delay_samples=1),
        "coefficient_mismatch": perturb_secondary_path_model(
            reloaded_model.impulse_response,
            coefficient_perturbation=perturbation,
        ),
        "phase_shaping_mismatch": perturb_secondary_path_model(
            reloaded_model.impulse_response,
            coefficient_perturbation=np.roll(perturbation, 1),
        ),
    }
    mismatch_runs = {
        name: run_anc_experiment(
            anc_reference,
            primary_path,
            secondary_path_true,
            path_model,
            anc_config,
        )
        for name, path_model in models.items()
    }
    mismatch_metrics = {
        name: {
            **anc_metrics(run, no_control),
            "final_power_vs_identified_model": float(
                run.final_error_power
                / (
                    mismatch_runs["identified_perfect_case"].final_error_power
                    + np.finfo(np.float64).eps
                )
            ),
        }
        for name, run in mismatch_runs.items()
    }
    save_line_plot(
        results_directory / "anc_mismatch_degradation.png",
        {name: run.error_power for name, run in mismatch_runs.items()},
        title="ANC error power degradation from secondary-path mismatch",
        ylabel="Error power",
        logarithmic_y=True,
    )

    initial_window = min(1_000, len(identification.squared_error) // 4)
    estimator_converged = float(np.mean(identification.squared_error[-initial_window:])) < float(
        np.mean(identification.squared_error[:initial_window])
    )
    acceptance = {
        "estimator_converges": estimator_converged,
        "low_impulse_response_error": validation.relative_impulse_error < 0.05,
        "model_artifact_reloads": bool(
            np.allclose(reloaded_model.impulse_response, model.impulse_response)
            and reloaded_model.sampling_rate_hz == sampling_rate_hz
        ),
        "identified_model_reduces_anc_error": (
            mismatch_metrics["identified_perfect_case"]["final_power_vs_no_control"] < 1.0
        ),
        "delay_mismatch_degrades_anc": (
            mismatch_metrics["delay_mismatch"]["final_power_vs_identified_model"] > 1.2
        ),
    }
    summary = {
        "module": "M5",
        "experiment": "Secondary-path identification and ANC mismatch validation",
        "identification_config": asdict(identification_config),
        "validation": validation_dict(validation),
        "model_artifact": str(model_path.relative_to(PROJECT_ROOT)),
        "model_metadata": model.to_metadata(),
        "mismatch_experiments": mismatch_metrics,
        "acceptance": acceptance,
        "status": "PASS" if all(acceptance.values()) else "FAIL",
    }
    with (results_directory / "secondary_path_summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)

    print(f"Module 5 secondary-path identification: {summary['status']}")
    print(f"Relative impulse-response error: {validation.relative_impulse_error:.3e}")
    print(f"Results saved to: {results_directory}")


if __name__ == "__main__":
    main()
