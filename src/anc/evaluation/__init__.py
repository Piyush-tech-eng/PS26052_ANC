"""Module 7 standardized evaluation and AI-ready dataset interfaces."""

from anc.evaluation.baselines import BaselineEntry, build_benchmark_manifest, save_baseline_package, save_metrics_table
from anc.evaluation.dataset import DatasetWindow, WindowConfig, compute_normalization, extract_windows, load_dataset, save_dataset, split_dataset
from anc.evaluation.metrics import compute_band_attenuation, compute_convergence_profile, compute_noise_reduction_db, compute_pesq_approx, compute_segment_mse, compute_si_snr, compute_signal_power, compute_stoi, detect_divergence, evaluate_scenario
from anc.evaluation.scenarios import ScenarioDefinition, ScenarioResult, build_scenario_matrix, run_scenario

__all__ = ["BaselineEntry", "DatasetWindow", "ScenarioDefinition", "ScenarioResult", "WindowConfig", "build_benchmark_manifest", "build_scenario_matrix", "compute_band_attenuation", "compute_convergence_profile", "compute_noise_reduction_db", "compute_normalization", "compute_pesq_approx", "compute_segment_mse", "compute_si_snr", "compute_signal_power", "compute_stoi", "detect_divergence", "evaluate_scenario", "extract_windows", "load_dataset", "run_scenario", "save_baseline_package", "save_dataset", "save_metrics_table", "split_dataset"]

