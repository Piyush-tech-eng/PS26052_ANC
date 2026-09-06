from __future__ import annotations

import numpy as np

from anc.evaluation.dataset import DatasetWindow, WindowConfig, compute_normalization, load_dataset, save_dataset, split_dataset


def _window(run_id: str, index: int) -> DatasetWindow:
    values = np.full(8, index, dtype=np.float64)
    return DatasetWindow(values, values * 2, values, values, None, 8_000, "scenario", run_id, f"{run_id}-{index}", {"controller": "fxnlms"}, {}, "synthetic")


def test_split_is_run_level_and_round_trips(tmp_path) -> None:
    windows = [_window(run_id, index) for index, run_id in enumerate(("run-a", "run-a", "run-b", "run-c", "run-d"))]
    splits = split_dataset(windows)
    assigned = {split: {window.run_id for window in values} for split, values in splits.items()}
    assert not assigned["train"] & assigned["validation"]
    assert not assigned["train"] & assigned["test"]
    normalization = compute_normalization(splits["train"])
    save_dataset(tmp_path, splits, normalization, WindowConfig(8, 4))
    reloaded, reloaded_normalization = load_dataset(tmp_path)
    assert sum(map(len, reloaded.values())) == len(windows)
    assert reloaded_normalization == normalization
