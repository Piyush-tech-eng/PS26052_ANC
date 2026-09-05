"""Module 4 reproducible digital feedforward ANC plant acceptance experiment."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anc.plant import (
    ANCExperimentConfig,
    apply_causal_fir,
    make_delayed_fir_path,
    run_anc_experiment,
)
from anc.visualization import save_line_plot


def result_metrics(result, no_control) -> dict[str, float]:
    return {
        "initial_error_power": result.initial_error_power,
        "final_error_power": result.final_error_power,
        "within_run_error_power_ratio": result.residual_power_ratio,
        "final_power_vs_no_control": float(
            result.final_error_power
            / (no_control.final_error_power + np.finfo(np.float64).eps)
        ),
        "attenuation_db_vs_no_control": float(
            10.0
            * np.log10(
                (no_control.final_error_power + np.finfo(np.float64).eps)
                / (result.final_error_power + np.finfo(np.float64).eps)
            )
        ),
    }


def run_perfect_model_case(reference, primary_path, secondary_path, *, sampling_rate_hz):
    no_control = run_anc_experiment(
        reference,
        primary_path,
        secondary_path,
        secondary_path,
        ANCExperimentConfig(32, 0.25, "none", sampling_rate_hz=sampling_rate_hz),
    )
    controlled = run_anc_experiment(
        reference,
        primary_path,
        secondary_path,
        secondary_path,
        ANCExperimentConfig(32, 0.25, "fxnlms", sampling_rate_hz=sampling_rate_hz),
    )
    return result_metrics(controlled, no_control)


def main() -> None:
    results_directory = PROJECT_ROOT / "results" / "m4_01_digital_anc_plant"
    results_directory.mkdir(parents=True, exist_ok=True)

    sampling_rate_hz = 8_000
    rng = np.random.default_rng(26052)
    reference = rng.normal(0.0, 1.0, 18_000)

    # Delayed, phase-shaping FIR paths make the secondary path a genuine part
    # of the controller-to-error relationship, not a direct-output shortcut.
    primary_path = make_delayed_fir_path(
        [0.85, 0.25, -0.10],
        delay_samples=2,
    )
    secondary_path = make_delayed_fir_path(
        [0.55, 0.18, -0.08],
        delay_samples=2,
    )

    algorithm_configs = {
        "no_control": ANCExperimentConfig(32, 0.25, "none", sampling_rate_hz=sampling_rate_hz),
        "lms_baseline": ANCExperimentConfig(32, 0.005, "lms", sampling_rate_hz=sampling_rate_hz),
        "nlms_baseline": ANCExperimentConfig(32, 0.10, "nlms", sampling_rate_hz=sampling_rate_hz),
        "fxlms": ANCExperimentConfig(32, 0.005, "fxlms", sampling_rate_hz=sampling_rate_hz),
        "fxnlms": ANCExperimentConfig(32, 0.25, "fxnlms", sampling_rate_hz=sampling_rate_hz),
    }
    runs = {
        name: run_anc_experiment(
            reference,
            primary_path,
            secondary_path,
            secondary_path,
            config,
        )
        for name, config in algorithm_configs.items()
    }

    no_control = runs["no_control"]
    controlled = runs["fxnlms"]

    # Preserve the exact required Module 4 artifact names for the canonical
    # perfect-model FxNLMS experiment.
    np.save(results_directory / "reference_x.npy", controlled.reference)
    np.save(results_directory / "primary_disturbance_d.npy", controlled.primary_disturbance)
    np.save(results_directory / "controller_output_y.npy", controlled.controller_output)
    np.save(results_directory / "secondary_path_output.npy", controlled.secondary_path_output)
    np.save(results_directory / "error_e.npy", controlled.error)
    np.save(results_directory / "controller_coefficients.npy", controlled.final_coefficients)
    np.save(results_directory / "controller_coefficient_history.npy", controlled.coefficient_history)
    np.save(results_directory / "primary_path.npy", controlled.primary_path)
    np.save(results_directory / "secondary_path_true.npy", controlled.secondary_path_true)
    np.save(results_directory / "secondary_path_model_used.npy", controlled.secondary_path_model_used)
    np.save(results_directory / "no_control_error.npy", no_control.error)

    for name, run in runs.items():
        algorithm_directory = results_directory / name
        algorithm_directory.mkdir(exist_ok=True)
        np.save(algorithm_directory / "error.npy", run.error)
        np.save(algorithm_directory / "error_power.npy", run.error_power)
        np.save(algorithm_directory / "controller_coefficients.npy", run.final_coefficients)

    # The required plant conditions use the same controller configuration but
    # deliberately change the reference statistics and path delay.
    colored_reference = apply_causal_fir(reference[:7_000], np.array([1.0, 0.7, 0.25]))
    tone_time = np.arange(7_000) / sampling_rate_hz
    tonal_reference = np.sin(2.0 * np.pi * 250.0 * tone_time) + 0.3 * np.sin(
        2.0 * np.pi * 670.0 * tone_time
    )
    delay_primary = make_delayed_fir_path([0.85, 0.25, -0.10], delay_samples=4)
    delay_secondary = make_delayed_fir_path([0.55, 0.18, -0.08], delay_samples=4)
    required_cases = {
        "stationary_broadband": result_metrics(controlled, no_control),
        "colored_disturbance": run_perfect_model_case(
            colored_reference,
            primary_path,
            secondary_path,
            sampling_rate_hz=sampling_rate_hz,
        ),
        "tonal_disturbance": run_perfect_model_case(
            tonal_reference,
            primary_path,
            secondary_path,
            sampling_rate_hz=sampling_rate_hz,
        ),
        "controlled_delay": run_perfect_model_case(
            reference[:7_000],
            delay_primary,
            delay_secondary,
            sampling_rate_hz=sampling_rate_hz,
        ),
    }

    save_line_plot(
        results_directory / "primary_disturbance_vs_residual_error.png",
        {"Primary disturbance": controlled.primary_disturbance, "Residual error": controlled.error},
        title="Primary disturbance and FxNLMS residual error",
        ylabel="Amplitude",
    )
    save_line_plot(
        results_directory / "no_control_vs_controlled_error.png",
        {"No control": no_control.error, "FxNLMS": controlled.error},
        title="No-control baseline versus filtered-x control",
        ylabel="Error",
    )
    save_line_plot(
        results_directory / "controller_coefficient_trajectories.png",
        {
            f"w{index}": controlled.coefficient_history[:, index]
            for index in range(min(8, controlled.coefficient_history.shape[1]))
        },
        title="FxNLMS controller coefficient trajectories",
        ylabel="Controller coefficient",
    )
    save_line_plot(
        results_directory / "error_power_learning_curve.png",
        {name: run.error_power for name, run in runs.items()},
        title="ANC error power by algorithm",
        ylabel="Error power",
        logarithmic_y=True,
    )
    frequencies = np.fft.rfftfreq(len(reference), d=1.0 / sampling_rate_hz)
    save_line_plot(
        results_directory / "frequency_domain_attenuation.png",
        {
            "No control": np.abs(np.fft.rfft(no_control.error)),
            "FxNLMS residual": np.abs(np.fft.rfft(controlled.error)),
        },
        title="Frequency-domain residual attenuation",
        xlabel="Frequency (Hz)",
        ylabel="Magnitude",
        x_values=frequencies,
        logarithmic_y=True,
    )
    save_line_plot(
        results_directory / "algorithm_comparison.png",
        {name: run.error_power for name, run in runs.items()},
        title="No control, direct adaptation, and filtered-x comparison",
        ylabel="Error power",
        logarithmic_y=True,
    )

    algorithm_metrics = {
        name: {
            "config": asdict(algorithm_configs[name]),
            **result_metrics(run, no_control),
        }
        for name, run in runs.items()
    }
    error_identity = np.allclose(
        controlled.error,
        controlled.primary_disturbance + controlled.secondary_path_output,
        rtol=1e-12,
        atol=1e-12,
    )
    secondary_path_included = np.allclose(
        controlled.secondary_path_output,
        apply_causal_fir(controlled.controller_output, secondary_path),
        rtol=1e-12,
        atol=1e-12,
    )
    acceptance = {
        "error_signal_identity": bool(error_identity),
        "secondary_path_is_applied": bool(secondary_path_included),
        "no_control_baseline_saved": bool(np.any(np.abs(no_control.error) > 0.0)),
        "fxlms_reduces_error": runs["fxlms"].final_error_power < no_control.final_error_power,
        "fxnlms_reduces_error": controlled.final_error_power < no_control.final_error_power,
        "perfect_model_is_explicit": bool(np.array_equal(secondary_path, controlled.secondary_path_model_used)),
        "all_required_cases_reduce_error": all(
            values["final_power_vs_no_control"] < 1.0
            for values in required_cases.values()
        ),
    }
    summary = {
        "module": "M4",
        "experiment": "Digital feedforward ANC plant",
        "sign_convention": "d = P*x; y = W*x; y_s = S*y; e = d + y_s",
        "sampling_rate_hz": sampling_rate_hz,
        "primary_path_length": int(len(primary_path)),
        "secondary_path_length": int(len(secondary_path)),
        "algorithm_comparison": algorithm_metrics,
        "required_digital_anc_cases": required_cases,
        "acceptance": acceptance,
        "status": "PASS" if all(acceptance.values()) else "FAIL",
    }
    with (results_directory / "anc_summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)

    print(f"Module 4 digital ANC plant: {summary['status']}")
    print(f"FxNLMS attenuation: {algorithm_metrics['fxnlms']['attenuation_db_vs_no_control']:.2f} dB")
    print(f"Results saved to: {results_directory}")


if __name__ == "__main__":
    main()
