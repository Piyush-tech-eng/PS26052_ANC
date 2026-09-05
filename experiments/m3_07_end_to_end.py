"""Module 3 end-to-end adaptive-filter laboratory.

This experiment consolidates the earlier focused Module 3 scripts into one
configuration-driven, reproducible acceptance run.  It consumes the Module 1
handoff and compares online adaptation with the Module 2 Wiener benchmark.
"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from anc.adaptive import AdaptiveExperimentConfig, run_adaptive_experiment
from anc.visualization import save_line_plot


def causal_fir(signal: np.ndarray, impulse_response: np.ndarray) -> np.ndarray:
    return np.convolve(signal, impulse_response, mode="full")[: len(signal)]


def comparison_dict(result) -> dict[str, float | bool | int] | None:
    if result.wiener_comparison is None:
        return None
    comparison = result.wiener_comparison
    return {
        "initial_coefficient_error": comparison.initial_coefficient_error,
        "final_coefficient_error": comparison.final_coefficient_error,
        "coefficient_error_ratio": comparison.coefficient_error_ratio,
        "minimum_coefficient_error": comparison.minimum_coefficient_error,
        "minimum_error_index": comparison.minimum_error_index,
        "moved_closer_to_wiener": comparison.moved_closer_to_wiener,
    }


def result_summary(result) -> dict[str, object]:
    return {
        "config": asdict(result.config),
        "initial_error_power": result.convergence.initial_error_power,
        "final_error_power": result.convergence.final_error_power,
        "error_power_ratio": result.convergence.error_power_ratio,
        "error_decreased": result.convergence.error_decreased,
        "coefficients_stabilized": result.convergence.coefficients_stabilized,
        "final_coefficient_norm": float(np.linalg.norm(result.final_coefficients)),
        "wiener_comparison": comparison_dict(result),
    }


def main() -> None:
    results_directory = PROJECT_ROOT / "results" / "m3_07_end_to_end"
    results_directory.mkdir(parents=True, exist_ok=True)

    handoff_directory = PROJECT_ROOT / "results" / "m1_handoff"
    reference_path = handoff_directory / "reference.npy"
    desired_path = handoff_directory / "desired.npy"
    wiener_path = PROJECT_ROOT / "results" / "m2_06_wiener_solution" / "wiener_coefficients.npy"
    for source in (reference_path, desired_path, wiener_path):
        if not source.is_file():
            raise FileNotFoundError(f"Required upstream artifact was not found: {source}")

    # A fixed stationary subset makes the run quick while preserving the
    # Module 1/2 signal convention and 32-tap Wiener comparison.
    sample_count = 20_000
    reference = np.load(reference_path, allow_pickle=False)[:sample_count]
    desired = np.load(desired_path, allow_pickle=False)[:sample_count]
    wiener_coefficients = np.load(wiener_path, allow_pickle=False)

    lms_config = AdaptiveExperimentConfig(
        filter_length=len(wiener_coefficients),
        step_size=0.01,
        algorithm="lms",
        sampling_rate_hz=8_000,
        experiment_seed=26052,
        learning_window=256,
    )
    nlms_config = AdaptiveExperimentConfig(
        filter_length=len(wiener_coefficients),
        step_size=0.4,
        algorithm="nlms",
        epsilon=1e-8,
        sampling_rate_hz=8_000,
        experiment_seed=26052,
        learning_window=256,
    )

    lms = run_adaptive_experiment(
        reference,
        desired,
        lms_config,
        wiener_coefficients=wiener_coefficients,
    )
    nlms = run_adaptive_experiment(
        reference,
        desired,
        nlms_config,
        wiener_coefficients=wiener_coefficients,
    )

    # The required artifact names describe the canonical LMS run.  Both
    # algorithms are also stored in named subdirectories for direct reuse.
    np.save(results_directory / "adaptive_output.npy", lms.output)
    np.save(results_directory / "error.npy", lms.error)
    np.save(results_directory / "final_coefficients.npy", lms.final_coefficients)
    np.save(results_directory / "coefficient_history.npy", lms.coefficient_history)
    np.save(results_directory / "instantaneous_squared_error.npy", lms.squared_error)
    np.save(results_directory / "mse_learning_curve.npy", lms.mse_learning_curve)

    for algorithm_name, result in (("lms", lms), ("nlms", nlms)):
        algorithm_directory = results_directory / algorithm_name
        algorithm_directory.mkdir(exist_ok=True)
        np.save(algorithm_directory / "output.npy", result.output)
        np.save(algorithm_directory / "error.npy", result.error)
        np.save(algorithm_directory / "squared_error.npy", result.squared_error)
        np.save(algorithm_directory / "coefficient_history.npy", result.coefficient_history)
        np.save(algorithm_directory / "final_coefficients.npy", result.final_coefficients)
        np.save(algorithm_directory / "mse_learning_curve.npy", result.mse_learning_curve)

    # Required Module 3 non-stationary cases: all use the same causal
    # identification path, but change the reference statistics deliberately.
    rng = np.random.default_rng(26052)
    case_samples = 6_000
    true_path = np.array([0.62, -0.22, 0.11, 0.05], dtype=np.float64)
    white = rng.normal(0.0, 1.0, case_samples)
    colored = causal_fir(white, np.array([1.0, 0.65, 0.25]))
    time = np.arange(case_samples) / lms_config.sampling_rate_hz
    tonal = np.sin(2.0 * np.pi * 280.0 * time) + 0.35 * np.sin(
        2.0 * np.pi * 640.0 * time
    )
    changing = np.concatenate((0.35 * white[: case_samples // 2], 2.5 * white[case_samples // 2 :]))
    cases = {
        "white_noise": white,
        "correlated_input": colored,
        "tonal_input": tonal,
        "changing_statistics": changing,
    }
    case_results = {}
    for name, case_reference in cases.items():
        case_result = run_adaptive_experiment(
            case_reference,
            causal_fir(case_reference, true_path),
            AdaptiveExperimentConfig(
                filter_length=16,
                step_size=0.003,
                algorithm="lms",
                sampling_rate_hz=lms_config.sampling_rate_hz,
                experiment_seed=26052,
                learning_window=128,
            ),
        )
        case_results[name] = result_summary(case_result)

    # NLMS must be less sensitive to a controlled input-power increase.
    scale = 5.0
    scale_reference = scale * white[:4_000]
    scale_desired = causal_fir(scale_reference, true_path)
    scaled_lms = run_adaptive_experiment(
        scale_reference,
        scale_desired,
        AdaptiveExperimentConfig(16, 0.005, "lms", learning_window=128),
    )
    scaled_nlms = run_adaptive_experiment(
        scale_reference,
        scale_desired,
        AdaptiveExperimentConfig(16, 0.4, "nlms", learning_window=128),
    )

    # Coefficient trajectories: plotting every tap is useful but unreadable,
    # so retain the first eight while the full numerical history is saved.
    save_line_plot(
        results_directory / "coefficient_trajectories.png",
        {
            f"w{coefficient_index}": lms.coefficient_history[:, coefficient_index]
            for coefficient_index in range(min(8, lms.coefficient_history.shape[1]))
        },
        title="Module 3 LMS coefficient trajectories",
        ylabel="Coefficient",
    )
    save_line_plot(
        results_directory / "instantaneous_squared_error.png",
        {"LMS": lms.squared_error, "NLMS": nlms.squared_error},
        title="Instantaneous squared error",
        ylabel="Squared error",
        logarithmic_y=True,
    )
    save_line_plot(
        results_directory / "mse_learning_curve.png",
        {"LMS": lms.mse_learning_curve, "NLMS": nlms.mse_learning_curve},
        title="MSE learning curves",
        ylabel="Causal MSE",
        logarithmic_y=True,
    )
    save_line_plot(
        results_directory / "lms_vs_nlms.png",
        {"LMS": lms.mse_learning_curve, "NLMS": nlms.mse_learning_curve},
        title="LMS vs NLMS on identical stationary data",
        ylabel="Causal MSE",
        logarithmic_y=True,
    )
    save_line_plot(
        results_directory / "wiener_comparison.png",
        {
            "LMS to Wiener": lms.wiener_comparison.coefficient_error_norm,
            "NLMS to Wiener": nlms.wiener_comparison.coefficient_error_norm,
        },
        title="Adaptive coefficient distance to Module 2 Wiener benchmark",
        ylabel="L2 coefficient error",
        logarithmic_y=True,
    )

    acceptance = {
        "lms_converged": lms.convergence.error_decreased,
        "nlms_converged": nlms.convergence.error_decreased,
        "lms_moves_toward_wiener": bool(lms.wiener_comparison.moved_closer_to_wiener),
        "nlms_moves_toward_wiener": bool(nlms.wiener_comparison.moved_closer_to_wiener),
        "nlms_less_scale_sensitive": (
            scaled_nlms.convergence.error_power_ratio
            < scaled_lms.convergence.error_power_ratio
        ),
        "all_case_outputs_finite": all(
            np.isfinite(value["final_error_power"])
            for value in case_results.values()
        ),
    }
    summary = {
        "module": "M3",
        "experiment": "Configuration-driven adaptive-filter laboratory",
        "input_artifacts": {
            "reference": str(reference_path.relative_to(PROJECT_ROOT)),
            "desired": str(desired_path.relative_to(PROJECT_ROOT)),
            "wiener_coefficients": str(wiener_path.relative_to(PROJECT_ROOT)),
        },
        "stationary_lms": result_summary(lms),
        "stationary_nlms": result_summary(nlms),
        "controlled_cases": case_results,
        "input_scale_robustness": {
            "scale_factor": scale,
            "lms": result_summary(scaled_lms),
            "nlms": result_summary(scaled_nlms),
        },
        "acceptance": acceptance,
        "status": "PASS" if all(acceptance.values()) else "FAIL",
    }
    with (results_directory / "summary.json").open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)

    print(f"Module 3 end-to-end laboratory: {summary['status']}")
    print(f"Results saved to: {results_directory}")


if __name__ == "__main__":
    main()
