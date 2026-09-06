"""Versioned, leakage-aware fixed-window dataset export for Module 8."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from anc.evaluation.scenarios import ScenarioResult


@dataclass(frozen=True)
class WindowConfig:
    window_length: int
    hop_length: int

    def __post_init__(self) -> None:
        if not isinstance(self.window_length, int) or self.window_length <= 0:
            raise ValueError("window_length must be a positive integer.")
        if not isinstance(self.hop_length, int) or self.hop_length <= 0:
            raise ValueError("hop_length must be a positive integer.")


@dataclass(frozen=True)
class DatasetWindow:
    reference_x: np.ndarray
    baseline_residual: np.ndarray
    controller_output_y: np.ndarray | None
    filtered_reference: np.ndarray | None
    target: np.ndarray | None
    sample_rate: int
    scenario_id: str
    run_id: str
    window_id: str
    algorithm_metadata: dict[str, object]
    path_metadata: dict[str, object]
    source_type: str

    def __post_init__(self) -> None:
        length = len(np.asarray(self.reference_x))
        if length == 0 or np.asarray(self.reference_x).ndim != 1:
            raise ValueError("reference_x must be a non-empty vector.")
        for name in ("baseline_residual", "controller_output_y", "filtered_reference", "target"):
            value = getattr(self, name)
            if value is not None and (np.asarray(value).ndim != 1 or len(value) != length):
                raise ValueError(f"{name} must be None or match reference_x length.")
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        if self.source_type not in {"synthetic", "recorded"}:
            raise ValueError("source_type must be 'synthetic' or 'recorded'.")


def extract_windows(results: Iterable[ScenarioResult], config: WindowConfig) -> list[DatasetWindow]:
    """Slice completed runs into full fixed-length windows with explicit availability."""

    windows: list[DatasetWindow] = []
    for result in results:
        replay = result.replay_result
        for start in range(0, len(replay.residual) - config.window_length + 1, config.hop_length):
            end = start + config.window_length
            index = start // config.hop_length
            windows.append(DatasetWindow(
                reference_x=replay.reference[start:end].copy(), baseline_residual=replay.residual[start:end].copy(),
                controller_output_y=replay.controller_output[start:end].copy(), filtered_reference=replay.filtered_reference[start:end].copy(),
                target=None if result.target is None else result.target[start:end].copy(), sample_rate=replay.sampling_rate_hz,
                scenario_id=result.definition.scenario_id, run_id=result.run_id, window_id=f"{result.run_id}__w{index:05d}",
                algorithm_metadata={"controller": result.definition.controller, **result.definition.config_snapshot},
                path_metadata={"secondary_path_model": result.definition.secondary_path_model, "path_condition": result.definition.path_condition},
                source_type=result.definition.source_type,
            ))
    return windows


def split_dataset(windows: Iterable[DatasetWindow], *, train_fraction: float = 0.6, validation_fraction: float = 0.2) -> dict[str, list[DatasetWindow]]:
    """Split whole runs (never individual windows) deterministically to avoid leakage."""

    all_windows = list(windows)
    groups: dict[str, list[DatasetWindow]] = {}
    for window in all_windows:
        groups.setdefault(window.run_id, []).append(window)
    if len(groups) < 3:
        raise ValueError("At least three complete runs are required for train/validation/test splitting.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave room for a test split.")
    run_ids = sorted(groups, key=lambda run_id: hashlib.sha256(run_id.encode("utf-8")).hexdigest())
    count = len(run_ids)
    train_count = max(1, int(np.floor(count * train_fraction)))
    validation_count = max(1, int(np.floor(count * validation_fraction)))
    if train_count + validation_count >= count:
        train_count = count - 2
        validation_count = 1
    partitions = {"train": run_ids[:train_count], "validation": run_ids[train_count:train_count + validation_count], "test": run_ids[train_count + validation_count:]}
    return {name: [window for run_id in run_ids for window in groups[run_id] if run_id in selected] for name, selected in partitions.items()}


def compute_normalization(training_windows: Iterable[DatasetWindow]) -> dict[str, object]:
    """Compute reusable z-score statistics from training windows only."""

    windows = list(training_windows)
    if not windows:
        raise ValueError("At least one training window is required.")
    fields = {"reference_x": [window.reference_x for window in windows], "baseline_residual": [window.baseline_residual for window in windows]}
    statistics: dict[str, dict[str, float]] = {}
    for name, arrays in fields.items():
        joined = np.concatenate(arrays)
        statistics[name] = {"mean": float(np.mean(joined)), "std": float(max(np.std(joined), np.finfo(np.float64).eps))}
    return {"format_version": 1, "policy": "z-score; statistics derived from train split only", "fields": statistics}


def _metadata(window: DatasetWindow) -> dict[str, object]:
    return {"sample_rate": window.sample_rate, "scenario_id": window.scenario_id, "run_id": window.run_id, "window_id": window.window_id,
            "algorithm_metadata": window.algorithm_metadata, "path_metadata": window.path_metadata, "source_type": window.source_type}


def save_dataset(output_dir: str | Path, splits: dict[str, list[DatasetWindow]], normalization: dict[str, object], config: WindowConfig) -> Path:
    """Save loadable `.npz` windows, a split manifest, and training-only normalization."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    memberships: dict[str, list[dict[str, object]]] = {}
    for split, windows in splits.items():
        destination = root / "windows" / split
        destination.mkdir(parents=True, exist_ok=True)
        memberships[split] = []
        for window in windows:
            path = destination / f"{window.window_id}.npz"
            np.savez_compressed(path, reference_x=window.reference_x, baseline_residual=window.baseline_residual,
                                controller_output_y=np.array([]) if window.controller_output_y is None else window.controller_output_y,
                                filtered_reference=np.array([]) if window.filtered_reference is None else window.filtered_reference,
                                target=np.array([]) if window.target is None else window.target,
                                controller_output_available=np.asarray(window.controller_output_y is not None),
                                filtered_reference_available=np.asarray(window.filtered_reference is not None), target_available=np.asarray(window.target is not None),
                                metadata_json=np.asarray(json.dumps(_metadata(window), sort_keys=True)))
            memberships[split].append({**_metadata(window), "path": str(path.relative_to(root))})
    manifest = {"format": "ps26052-anc-ai-ready-dataset", "format_version": 1, "window_config": asdict(config), "splits": memberships}
    (root / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (root / "normalization.json").write_text(json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return root


def load_dataset(dataset_dir: str | Path) -> tuple[dict[str, list[DatasetWindow]], dict[str, object]]:
    """Reload a dataset without importing any Module 6 experiment code."""

    root = Path(dataset_dir)
    manifest = json.loads((root / "dataset_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "ps26052-anc-ai-ready-dataset":
        raise ValueError("Unsupported dataset format.")
    output: dict[str, list[DatasetWindow]] = {}
    for split, entries in manifest["splits"].items():
        output[split] = []
        for entry in entries:
            with np.load(root / entry["path"], allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
                optional = lambda name, present: archive[name].copy() if bool(archive[present].item()) else None
                output[split].append(DatasetWindow(archive["reference_x"].copy(), archive["baseline_residual"].copy(),
                    optional("controller_output_y", "controller_output_available"), optional("filtered_reference", "filtered_reference_available"), optional("target", "target_available"),
                    metadata["sample_rate"], metadata["scenario_id"], metadata["run_id"], metadata["window_id"], metadata["algorithm_metadata"], metadata["path_metadata"], metadata["source_type"]))
    normalization = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    return output, normalization
