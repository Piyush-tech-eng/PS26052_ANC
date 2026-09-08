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
    # --- NEW speech-domain fields (Section 3.5 of AI plan) ---
    noisy_speech: np.ndarray | None = None
    clean_speech_target: np.ndarray | None = None
    noise_component: np.ndarray | None = None
    speech_metadata: dict[str, object] | None = None

    def __post_init__(self) -> None:
        length = len(np.asarray(self.reference_x))
        if length == 0 or np.asarray(self.reference_x).ndim != 1:
            raise ValueError("reference_x must be a non-empty vector.")
        for name in ("baseline_residual", "controller_output_y", "filtered_reference", "target",
                     "noisy_speech", "clean_speech_target", "noise_component"):
            value = getattr(self, name)
            if value is not None and (np.asarray(value).ndim != 1 or len(value) != length):
                raise ValueError(f"{name} must be None or match reference_x length.")
        if not isinstance(self.sample_rate, int) or self.sample_rate <= 0:
            raise ValueError("sample_rate must be a positive integer.")
        if self.source_type not in {"synthetic", "recorded", "speech"}:
            raise ValueError("source_type must be 'synthetic', 'recorded', or 'speech'.")


def extract_windows(results: Iterable[ScenarioResult], config: WindowConfig) -> list[DatasetWindow]:
    """Slice completed runs into full fixed-length windows with explicit availability."""

    windows: list[DatasetWindow] = []
    for result in results:
        replay = result.replay_result
        for start in range(0, len(replay.residual) - config.window_length + 1, config.hop_length):
            end = start + config.window_length
            index = start // config.hop_length
            speech_meta = None
            if getattr(result.definition, "speech_source_id", None) is not None:
                speech_meta = {
                    "speech_source_id": result.definition.speech_source_id,
                    "noise_family": result.definition.noise_family,
                    "snr_db": result.definition.snr_db,
                }
            windows.append(DatasetWindow(
                reference_x=replay.reference[start:end].copy(), baseline_residual=replay.residual[start:end].copy(),
                controller_output_y=replay.controller_output[start:end].copy(), filtered_reference=replay.filtered_reference[start:end].copy(),
                target=None if result.target is None else result.target[start:end].copy(), sample_rate=replay.sampling_rate_hz,
                scenario_id=result.definition.scenario_id, run_id=result.run_id, window_id=f"{result.run_id}__w{index:05d}",
                algorithm_metadata={"controller": result.definition.controller, **result.definition.config_snapshot},
                path_metadata={"secondary_path_model": result.definition.secondary_path_model, "path_condition": result.definition.path_condition},
                source_type=result.definition.source_type,
                speech_metadata=speech_meta,
            ))
    return windows


def split_dataset(windows: Iterable[DatasetWindow], *, train_fraction: float = 0.6, validation_fraction: float = 0.2) -> dict[str, list[DatasetWindow]]:
    """Split whole scenarios (never individual windows) deterministically to avoid leakage.

    Groups by ``scenario_id`` (not ``run_id``) so that all controller variants
    derived from the same underlying disturbance realization remain within the
    same split.
    """

    all_windows = list(windows)
    groups: dict[str, list[DatasetWindow]] = {}
    
    # First pass: identify speakers and noise families
    # A group key is speech_source_id if available, else scenario_id
    for window in all_windows:
        is_speech = window.speech_metadata is not None
        if is_speech:
            group_key = str(window.speech_metadata.get("speech_source_id", window.scenario_id))
        else:
            group_key = window.scenario_id
        groups.setdefault(group_key, []).append(window)

    if len(groups) < 3:
        raise ValueError("At least three distinct scenarios or speakers are required for train/validation/test splitting.")
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1 or train_fraction + validation_fraction >= 1:
        raise ValueError("Split fractions must be positive and leave room for a test split.")

    # Sort group keys for determinism
    group_keys = sorted(groups.keys(), key=lambda k: hashlib.sha256(k.encode("utf-8")).hexdigest())

    # Identify all noise families across the dataset
    all_noise_fams = set()
    for window in all_windows:
        if window.speech_metadata is not None and "noise_family" in window.speech_metadata:
            all_noise_fams.add(str(window.speech_metadata["noise_family"]))
            
    test_noise_fams = set()
    if len(all_noise_fams) >= 2:
        sorted_noise = sorted(list(all_noise_fams), key=lambda x: hashlib.sha256(x.encode()).hexdigest())
        test_noise_fams.add(sorted_noise[-1]) # Reserve exactly 1 noise family for test

    count = len(group_keys)
    train_count = max(1, int(np.floor(count * train_fraction)))
    validation_count = max(1, int(np.floor(count * validation_fraction)))
    
    # Ensure at least 2 in test if possible
    if count >= 4 and (count - train_count - validation_count) < 2:
        train_count = count - 3
        validation_count = 1

    train_keys = set(group_keys[:train_count])
    val_keys = set(group_keys[train_count:train_count + validation_count])
    test_keys = set(group_keys[train_count + validation_count:])
    
    # If there are fewer than 3 groups, the first check handles it.
    
    partitions = {"train": [], "validation": [], "test": []}
    
    for gk in group_keys:
        if gk in train_keys:
            target_split = "train"
        elif gk in val_keys:
            target_split = "validation"
        else:
            target_split = "test"
            
        for window in groups[gk]:
            # Enforce test noise family isolation
            # If this is train/val, we must drop any window that uses the test-only noise family.
            # We don't move it to test, because that would split the speaker.
            if target_split in {"train", "validation"}:
                if window.speech_metadata is not None:
                    noise = str(window.speech_metadata.get("noise_family", "unknown"))
                    if noise in test_noise_fams:
                        continue # Drop from train/val
            
            partitions[target_split].append(window)

    return partitions


def compute_normalization(training_windows: Iterable[DatasetWindow]) -> dict[str, object]:
    """Compute reusable z-score statistics from training windows only.

    Also covers speech-domain fields (noisy_speech, clean_speech_target)
    when they are present in the training windows.
    """

    windows = list(training_windows)
    if not windows:
        raise ValueError("At least one training window is required.")
    fields: dict[str, list[np.ndarray]] = {
        "reference_x": [window.reference_x for window in windows],
        "baseline_residual": [window.baseline_residual for window in windows],
    }
    # Include speech-domain fields if present
    noisy = [window.noisy_speech for window in windows if window.noisy_speech is not None]
    if noisy:
        fields["noisy_speech"] = noisy
    targets = [window.clean_speech_target for window in windows if window.clean_speech_target is not None]
    if targets:
        fields["clean_speech_target"] = targets
    statistics: dict[str, dict[str, float]] = {}
    for name, arrays in fields.items():
        joined = np.concatenate(arrays)
        statistics[name] = {"mean": float(np.mean(joined)), "std": float(max(np.std(joined), np.finfo(np.float64).eps))}
    return {"format_version": 1, "policy": "z-score; statistics derived from train split only", "fields": statistics}


def apply_normalization(
    window: DatasetWindow,
    normalization: dict[str, object],
) -> DatasetWindow:
    """Apply z-score normalization to a DatasetWindow's arrays.

    Uses the statistics computed by ``compute_normalization()`` (derived
    from the training split only) to normalize the window's signal arrays.
    This ensures that validation and test data are normalized using
    training-set statistics — preventing data leakage.

    Parameters
    ----------
    window : DatasetWindow
        The window to normalize.
    normalization : dict
        Normalization statistics from ``compute_normalization()``.

    Returns
    -------
    DatasetWindow
        A new DatasetWindow with normalized arrays.
    """
    fields_stats = normalization.get("fields", {})

    def _normalize(arr: np.ndarray | None, field_name: str) -> np.ndarray | None:
        if arr is None:
            return None
        stats = fields_stats.get(field_name)
        if stats is None:
            return arr.copy()
        mean = stats["mean"]
        std = stats["std"]
        return ((arr - mean) / std).astype(np.float64)

    return DatasetWindow(
        reference_x=_normalize(window.reference_x, "reference_x"),
        baseline_residual=_normalize(window.baseline_residual, "baseline_residual"),
        controller_output_y=window.controller_output_y.copy() if window.controller_output_y is not None else None,
        filtered_reference=window.filtered_reference.copy() if window.filtered_reference is not None else None,
        target=window.target.copy() if window.target is not None else None,
        sample_rate=window.sample_rate,
        scenario_id=window.scenario_id,
        run_id=window.run_id,
        window_id=window.window_id,
        algorithm_metadata=dict(window.algorithm_metadata),
        path_metadata=dict(window.path_metadata),
        source_type=window.source_type,
        noisy_speech=_normalize(window.noisy_speech, "noisy_speech"),
        clean_speech_target=_normalize(window.clean_speech_target, "clean_speech_target"),
        noise_component=window.noise_component.copy() if window.noise_component is not None else None,
        speech_metadata=dict(window.speech_metadata) if window.speech_metadata is not None else None,
    )


def _metadata(window: DatasetWindow) -> dict[str, object]:
    meta = {"sample_rate": window.sample_rate, "scenario_id": window.scenario_id, "run_id": window.run_id, "window_id": window.window_id,
            "algorithm_metadata": window.algorithm_metadata, "path_metadata": window.path_metadata, "source_type": window.source_type}
    if window.speech_metadata is not None:
        meta["speech_metadata"] = window.speech_metadata
    return meta


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
            arrays = dict(
                reference_x=window.reference_x, baseline_residual=window.baseline_residual,
                controller_output_y=np.array([]) if window.controller_output_y is None else window.controller_output_y,
                filtered_reference=np.array([]) if window.filtered_reference is None else window.filtered_reference,
                target=np.array([]) if window.target is None else window.target,
                controller_output_available=np.asarray(window.controller_output_y is not None),
                filtered_reference_available=np.asarray(window.filtered_reference is not None),
                target_available=np.asarray(window.target is not None),
                # Speech-domain arrays
                noisy_speech=np.array([]) if window.noisy_speech is None else window.noisy_speech,
                clean_speech_target=np.array([]) if window.clean_speech_target is None else window.clean_speech_target,
                noise_component=np.array([]) if window.noise_component is None else window.noise_component,
                noisy_speech_available=np.asarray(window.noisy_speech is not None),
                clean_speech_target_available=np.asarray(window.clean_speech_target is not None),
                noise_component_available=np.asarray(window.noise_component is not None),
                metadata_json=np.asarray(json.dumps(_metadata(window), sort_keys=True)),
            )
            np.savez_compressed(path, **arrays)
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
                optional = lambda name, present: archive[name].copy() if (present in archive and bool(archive[present].item())) else None
                output[split].append(DatasetWindow(
                    reference_x=archive["reference_x"].copy(),
                    baseline_residual=archive["baseline_residual"].copy(),
                    controller_output_y=optional("controller_output_y", "controller_output_available"),
                    filtered_reference=optional("filtered_reference", "filtered_reference_available"),
                    target=optional("target", "target_available"),
                    sample_rate=metadata["sample_rate"],
                    scenario_id=metadata["scenario_id"],
                    run_id=metadata["run_id"],
                    window_id=metadata["window_id"],
                    algorithm_metadata=metadata["algorithm_metadata"],
                    path_metadata=metadata["path_metadata"],
                    source_type=metadata["source_type"],
                    noisy_speech=optional("noisy_speech", "noisy_speech_available"),
                    clean_speech_target=optional("clean_speech_target", "clean_speech_target_available"),
                    noise_component=optional("noise_component", "noise_component_available"),
                    speech_metadata=metadata.get("speech_metadata"),
                ))
    normalization = json.loads((root / "normalization.json").read_text(encoding="utf-8"))
    return output, normalization
