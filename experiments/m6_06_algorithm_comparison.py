"""M6.06: comparable no-control, FxLMS, and FxNLMS replay evidence.

The experiment consumes the portable Module 5 ``SecondaryPathModel`` rather
than importing any identification experiment. Its callable entry point makes
the required Module 6 artifact set testable without CLI or plotting globals.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from anc.io import prepare_replay_inputs, recording_from_array
from anc.plant import ANCExperimentConfig, apply_causal_fir
from anc.replay import run_anc_replay
from anc.secondary_path import SecondaryPathModel


def _reference_signal(rng: np.random.Generator, samples: int) -> np.ndarray:
    """Build a reproducible correlated reference representative of replay data."""

    white = rng.normal(0.0, 0.25, samples)
    return np.convolve(white, np.array([0.12, 0.24, 0.40, 0.24]), mode="same")


def _metrics(result: Any, no_control: Any) -> dict[str, float]:
    floor = np.finfo(np.float64).eps
    baseline_power = float(np.mean(no_control.residual**2))
    residual_power = float(np.mean(result.residual**2))
    return {
        "initial_residual_power": result.initial_residual_power,
        "final_residual_power": result.final_residual_power,
        "full_run_residual_power": residual_power,
        "attenuation_db_vs_no_control": float(10.0 * np.log10((baseline_power + floor) / (residual_power + floor))),
        "adaptation_attenuation_db": result.attenuation_db,
    }


def _save_plots(results: dict[str, Any], sample_rate: int, output_dir: Path) -> None:
    """Save time, power, coefficient, and spectrum diagnostics required by M6."""

    import matplotlib.pyplot as plt

    labels = {"none": "No control", "fxlms": "FxLMS", "fxnlms": "FxNLMS"}
    colours = {"none": "#666666", "fxlms": "#e69f00", "fxnlms": "#0072b2"}
    shown = min(4_000, len(results["none"].residual))
    time = np.arange(shown) / sample_rate
    figure, axis = plt.subplots(figsize=(12, 5))
    for name, result in results.items():
        axis.plot(time, result.residual[:shown], label=labels[name], color=colours[name], alpha=0.8)
    axis.set(xlabel="Time (s)", ylabel="Residual amplitude", title="M6.06 residual comparison")
    axis.legend(); axis.grid(alpha=0.3); figure.tight_layout(); figure.savefig(output_dir / "residual_waveforms.png", dpi=150); plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.plot(time, results["fxnlms"].reference[:shown], label="Controller reference", color="#009e73", alpha=0.7)
    axis.plot(time, results["fxnlms"].residual[:shown], label="FxNLMS residual", color=colours["fxnlms"], alpha=0.8)
    axis.set(xlabel="Time (s)", ylabel="Amplitude", title="M6.06 reference and controlled residual")
    axis.legend(); axis.grid(alpha=0.3); figure.tight_layout(); figure.savefig(output_dir / "reference_and_residual.png", dpi=150); plt.close(figure)

    time = np.arange(len(results["none"].residual_power)) / sample_rate
    figure, axis = plt.subplots(figsize=(12, 5))
    for name, result in results.items():
        axis.plot(time, result.residual_power, label=labels[name], color=colours[name])
    axis.set_yscale("log"); axis.set(xlabel="Time (s)", ylabel="Moving residual power", title="M6.06 residual-power convergence")
    axis.legend(); axis.grid(alpha=0.3); figure.tight_layout(); figure.savefig(output_dir / "residual_power.png", dpi=150); plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    for name in ("fxlms", "fxnlms"):
        axis.plot(np.linalg.norm(results[name].coefficient_history, axis=1), label=labels[name], color=colours[name])
    axis.set(xlabel="Sample", ylabel="Coefficient L2 norm", title="M6.06 controller trajectories")
    axis.legend(); axis.grid(alpha=0.3); figure.tight_layout(); figure.savefig(output_dir / "controller_coefficient_norm.png", dpi=150); plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 5))
    for name, result in results.items():
        residual = result.residual[-max(256, len(result.residual) // 4):]
        frequencies = np.fft.rfftfreq(len(residual), 1.0 / sample_rate)
        spectrum = 20.0 * np.log10(np.maximum(np.abs(np.fft.rfft(residual)), np.finfo(np.float64).eps))
        axis.plot(frequencies, spectrum, label=labels[name], color=colours[name])
    axis.set(xlabel="Frequency (Hz)", ylabel="Magnitude (dB)", title="M6.06 steady-state residual spectrum")
    axis.legend(); axis.grid(alpha=0.3); figure.tight_layout(); figure.savefig(output_dir / "residual_spectrum.png", dpi=150); plt.close(figure)


def run_algorithm_comparison(
    output_dir: str | Path, identified_model_path: str | Path, *, sampling_rate_hz: int = 8_000,
    duration_seconds: float = 4.0, seed: int = 26_052, create_plots: bool = True,
) -> dict[str, object]:
    """Run and save the complete Module 6 comparison artifact set."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive.")
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    model_path = Path(identified_model_path)
    model = SecondaryPathModel.load(model_path)
    if model.sampling_rate_hz != sampling_rate_hz:
        raise ValueError("identified model sampling rate must match the replay sampling rate.")
    rng = np.random.default_rng(seed)
    count = int(round(sampling_rate_hz * duration_seconds))
    reference = _reference_signal(rng, count)
    primary_path = np.array([0.0, 0.0, 0.75, 0.30, -0.12, 0.05])
    true_secondary_path = np.array([0.0, 0.0, 0.62, 0.28, -0.10, 0.04])
    disturbance = apply_causal_fir(reference, primary_path) + rng.normal(0.0, 0.001, count)
    inputs = prepare_replay_inputs(
        recording_from_array(reference, sampling_rate_hz=sampling_rate_hz, role="reference", source="synthetic:m6_06"),
        recording_from_array(disturbance, sampling_rate_hz=sampling_rate_hz, role="measured", source="synthetic:m6_06"), align=False,
    )
    configurations = {
        "none": ANCExperimentConfig(32, 0.05, "none", sampling_rate_hz=sampling_rate_hz),
        "fxlms": ANCExperimentConfig(32, 0.01, "fxlms", sampling_rate_hz=sampling_rate_hz),
        "fxnlms": ANCExperimentConfig(32, 0.05, "fxnlms", sampling_rate_hz=sampling_rate_hz),
    }
    results = {name: run_anc_replay(inputs, true_secondary_path, model.impulse_response, config) for name, config in configurations.items()}
    np.save(output / "reference_x.npy", reference)
    np.save(output / "error_input_or_disturbance.npy", disturbance)
    model.save(output / "secondary_path_model_used.npz")
    for name, result in results.items():
        run_dir = output / "runs" / name; run_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(run_dir / "anc_replay_output.npz", reference_x=result.reference, residual_error_e=result.residual,
                            controller_output_y=result.controller_output, filtered_reference=result.filtered_reference,
                            controller_coefficients=result.final_coefficients, coefficient_history=result.coefficient_history)
    selected = results["fxnlms"]
    np.save(output / "controller_output_y.npy", selected.controller_output)
    np.save(output / "residual_error_e.npy", selected.residual)
    np.save(output / "filtered_reference.npy", selected.filtered_reference)
    np.save(output / "controller_coefficients.npy", selected.final_coefficients)
    np.save(output / "coefficient_history.npy", selected.coefficient_history)
    scenario_manifest = {
        "scenario_id": "m6-06-synthetic-comparison", "source_type": "synthetic", "sampling_rate_hz": sampling_rate_hz,
        "duration_seconds": duration_seconds, "seed": seed, "timing": {"alignment_applied": False, "delay_samples": None},
        "paths": {"identified_secondary_path_model": str(model_path), "true_secondary_path_available": True},
        "provenance": {"reference": "synthetic:m6_06", "disturbance": "controlled primary path"},
        "algorithms": {name: {"algorithm": config.algorithm, "filter_length": config.filter_length, "step_size": config.step_size} for name, config in configurations.items()},
    }
    (output / "scenario_manifest.json").write_text(json.dumps(scenario_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {"module": "M6.06", "scenario": scenario_manifest, "status": "complete",
               "algorithms": {name: _metrics(result, results["none"]) for name, result in results.items()}}
    (output / "anc_replay_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "m6_comparison_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if create_plots:
        _save_plots(results, sampling_rate_hz, output)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="M6.06 ANC algorithm comparison using the Module 5 model artifact.")
    parser.add_argument("--identified-model", type=Path, default=Path("results/m5_01_secondary_path_identification/secondary_path_model.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/m6_06_algorithm_comparison"))
    parser.add_argument("--duration", type=float, default=4.0)
    arguments = parser.parse_args(argv)
    run_algorithm_comparison(arguments.output_dir, arguments.identified_model, duration_seconds=arguments.duration)
    print(f"M6.06 artifacts saved to {arguments.output_dir}")


if __name__ == "__main__":
    main()
