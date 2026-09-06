"""M7.01: build a standardized ANC evaluation and AI-ready dataset package."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

import numpy as np

from anc.evaluation import (
    WindowConfig, build_scenario_matrix, compute_normalization, evaluate_scenario,
    extract_windows, run_scenario, save_baseline_package, save_dataset, split_dataset,
)
from anc.plant import ANCExperimentConfig, apply_causal_fir
from anc.scenarios import ANCScenario, recording_from_array
from anc.secondary_path import SecondaryPathModel


def _signal(signal_class: str, sample_count: int, sample_rate: int, seed: int, level: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if signal_class == "broadband":
        values = rng.normal(0.0, 0.22, sample_count)
    elif signal_class == "colored":
        values = np.convolve(rng.normal(0.0, 0.30, sample_count), np.array([0.15, 0.30, 0.40, 0.15]), mode="same")
    elif signal_class == "tonal":
        time = np.arange(sample_count) / sample_rate
        values = 0.20 * np.sin(2 * np.pi * 400 * time) + 0.12 * np.sin(2 * np.pi * 900 * time)
    elif signal_class == "burst":
        values = rng.normal(0.0, 0.24, sample_count) * (np.sin(2 * np.pi * 1.5 * np.arange(sample_count) / sample_rate) > 0.4)
    else:
        raise ValueError(f"Unsupported signal class: {signal_class}")
    gain = {"low": 0.5, "nominal": 1.0, "high": 1.5}[level]
    return np.asarray(gain * values, dtype=np.float64)


def _model_and_true_path(condition: str, sample_rate: int, identified_model: SecondaryPathModel) -> tuple[SecondaryPathModel, np.ndarray]:
    """Use the M5 model as the nominal hand-off, perturbing only controlled cases."""

    true_path = identified_model.impulse_response.copy()
    model_path = identified_model.impulse_response.copy()
    if condition == "delay-variation":
        true_path = np.concatenate((np.zeros(1), true_path))[:-1]
    elif condition == "gain-variation":
        true_path = 1.15 * true_path
    elif condition == "identified-mismatch":
        model_path = np.array([0.0, 0.0, 0.56, 0.32, -0.06, 0.02])
    elif condition != "nominal":
        raise ValueError(f"Unsupported path condition: {condition}")
    return SecondaryPathModel(model_path, sample_rate, metadata={"condition": condition, "origin": "m5-secondary-path-artifact", **identified_model.metadata}), true_path


def _plot_results(results, output_dir: Path, splits) -> None:
    """Write standard comparison diagnostics without making plots part of core APIs."""

    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"; plot_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[str, list] = {}
    for result in results:
        grouped.setdefault(result.definition.scenario_id, []).append(result)
    first_id, first_runs = next(iter(grouped.items()))
    figure, axis = plt.subplots(figsize=(11, 4))
    for result in first_runs:
        axis.plot(result.replay_result.residual[:2_000], label=result.definition.controller)
    axis.set(title=f"Residual comparison: {first_id}", xlabel="Sample", ylabel="Residual")
    axis.legend(); axis.grid(alpha=0.3); figure.tight_layout(); figure.savefig(plot_dir / "scenario_residual_comparison.png", dpi=150); plt.close(figure)
    rows = [(result.definition.controller, result.metrics["attenuation_db_vs_no_control"]) for result in results if result.metrics["attenuation_db_vs_no_control"] is not None]
    figure, axis = plt.subplots(figsize=(9, 4))
    algorithms = sorted(set(algorithm for algorithm, _ in rows))
    axis.boxplot([[value for algorithm, value in rows if algorithm == name] for name in algorithms], tick_labels=algorithms)
    axis.set(title="Noise reduction by algorithm", ylabel="Attenuation vs no-control (dB)")
    axis.grid(axis="y", alpha=0.3); figure.tight_layout(); figure.savefig(plot_dir / "attenuation_distribution.png", dpi=150); plt.close(figure)

    figure, axis = plt.subplots(figsize=(11, 4))
    for result in first_runs:
        profile = result.metrics["convergence_profile"]
        axis.plot(profile, label=result.definition.controller)
    axis.set_yscale("log")
    axis.set(title=f"Convergence profile: {first_id}", xlabel="Sample", ylabel="Moving residual power")
    axis.legend(); axis.grid(alpha=0.3); figure.tight_layout(); figure.savefig(plot_dir / "convergence_stability_summary.png", dpi=150); plt.close(figure)

    bands = sorted({band for result in results for band in result.metrics["band_attenuation_db"]})
    controlled = [result for result in results if result.definition.controller != "none"]
    figure, axis = plt.subplots(figsize=(10, 4))
    names = sorted({result.definition.controller for result in controlled})
    positions = np.arange(len(bands))
    width = 0.35
    for index, name in enumerate(names):
        values = [np.mean([result.metrics["band_attenuation_db"].get(band, np.nan) for result in controlled if result.definition.controller == name]) for band in bands]
        axis.bar(positions + (index - (len(names) - 1) / 2) * width, values, width=width, label=name)
    axis.set(title="Frequency-band attenuation summary", xlabel="Frequency band", ylabel="Attenuation vs no-control (dB)")
    axis.set_xticks(positions, bands); axis.legend(); axis.grid(axis="y", alpha=0.3); figure.tight_layout(); figure.savefig(plot_dir / "frequency_band_attenuation.png", dpi=150); plt.close(figure)

    split_names = list(splits)
    scenario_counts = [len({window.scenario_id for window in splits[name]}) for name in split_names]
    source_counts = [len({window.source_type for window in splits[name]}) for name in split_names]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(split_names, scenario_counts); axes[0].set(title="Dataset split coverage", ylabel="Unique scenarios")
    axes[1].bar(split_names, source_counts); axes[1].set(title="Dataset source-type coverage", ylabel="Unique source types")
    for axis in axes: axis.grid(axis="y", alpha=0.3)
    figure.tight_layout(); figure.savefig(plot_dir / "dataset_split_summary.png", dpi=150); plt.close(figure)


def build_evaluation_package(
    output_dir: str | Path, *, sample_count: int = 8_000, sample_rate_hz: int = 8_000,
    identified_model_path: str | Path | None = None, create_plots: bool = True,
) -> dict[str, object]:
    """Run a configuration-defined M7 matrix and persist the complete package."""

    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    default_model = Path(__file__).resolve().parents[1] / "results" / "m5_01_secondary_path_identification" / "secondary_path_model.npz"
    model_artifact = Path(identified_model_path) if identified_model_path is not None else default_model
    if not model_artifact.is_file():
        raise FileNotFoundError(f"Module 5 secondary-path artifact is required: {model_artifact}")
    identified_model = SecondaryPathModel.load(model_artifact)
    if identified_model.sampling_rate_hz != sample_rate_hz:
        raise ValueError("identified model sampling rate must match sample_rate_hz.")
    definitions = build_scenario_matrix(
        signal_classes=("broadband", "colored", "tonal", "burst"), input_levels=("nominal",),
        path_conditions=("nominal", "identified-mismatch"), source_types=("synthetic",),
        controllers=("none", "fxlms", "fxnlms"), secondary_path_models=("identified",),
        config_snapshot={"filter_length": 32, "learning_window": 256, "secondary_path_model_artifact": str(model_artifact)}, seed=26_052,
    )
    raw_results = []
    primary_path = np.array([0.0, 0.0, 0.75, 0.30, -0.12, 0.05])
    for definition in definitions:
        reference = _signal(definition.signal_class, sample_count, sample_rate_hz, definition.seed or 0, definition.input_level)
        model, true_path = _model_and_true_path(definition.path_condition, sample_rate_hz, identified_model)
        disturbance = apply_causal_fir(reference, primary_path)
        scenario = ANCScenario(definition.scenario_id, recording_from_array(reference, sampling_rate_hz=sample_rate_hz, source_type="synthetic", provenance={"seed": definition.seed}),
                               recording_from_array(disturbance, sampling_rate_hz=sample_rate_hz, source_type="synthetic"), model, true_path,
                               metadata={"clean_target_available": False})
        steps = {"none": 0.05, "fxlms": 0.01, "fxnlms": 0.05}
        config = ANCExperimentConfig(32, steps[definition.controller], definition.controller, sampling_rate_hz=sample_rate_hz, learning_window=256)
        raw_results.append(run_scenario(definition, scenario, config))
    by_scenario: dict[str, list] = {}
    for result in raw_results:
        by_scenario.setdefault(result.definition.scenario_id, []).append(result)
    results = []
    for runs in by_scenario.values():
        baseline = next(run for run in runs if run.definition.controller == "none").replay_result.residual
        for result in runs:
            metrics = evaluate_scenario(result.replay_result.residual, sampling_rate_hz=sample_rate_hz, baseline_residual=baseline,
                                        coefficient_history=result.replay_result.coefficient_history, convergence_window=256)
            results.append(replace(result, metrics=metrics))
    entries, _ = save_baseline_package(results, output)
    windows = extract_windows(results, WindowConfig(512, 256))
    splits = split_dataset(windows)
    normalization = compute_normalization(splits["train"])
    save_dataset(output, splits, normalization, WindowConfig(512, 256))
    summary = {"scenario_count": len(by_scenario), "run_count": len(results), "window_count": len(windows),
               "baseline_manifest": "benchmark_manifest.json", "dataset_manifest": "dataset_manifest.json",
               "algorithms": sorted({entry.algorithm for entry in entries})}
    (output / "m7_evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if create_plots:
        _plot_results(results, output, splits)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build the Module 7 standardized ANC benchmark and dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/m7_01_scenario_matrix_evaluation"))
    parser.add_argument("--sample-count", type=int, default=8_000)
    parser.add_argument("--identified-model", type=Path, default=None, help="Module 5 .npz secondary-path model artifact")
    arguments = parser.parse_args(argv)
    summary = build_evaluation_package(arguments.output_dir, sample_count=arguments.sample_count, identified_model_path=arguments.identified_model)
    print(f"M7 package complete: {summary['run_count']} runs, {summary['window_count']} windows")


if __name__ == "__main__":
    main()
