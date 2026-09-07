from __future__ import annotations

import numpy as np

from anc.evaluation.dataset import DatasetWindow, WindowConfig, compute_normalization, load_dataset, save_dataset, split_dataset


def _window(scenario_id: str, run_id: str, index: int) -> DatasetWindow:
    values = np.full(8, index, dtype=np.float64)
    return DatasetWindow(values, values * 2, values, values, None, 8_000, scenario_id, run_id, f"{run_id}-{index}", {"controller": "fxnlms"}, {}, "synthetic")


def test_split_is_scenario_level_and_round_trips(tmp_path) -> None:
    """Splitting groups by scenario_id; controller variants stay together."""
    windows = [
        _window("scenario-a", "scenario-a__fxlms", 0),
        _window("scenario-a", "scenario-a__fxnlms", 1),
        _window("scenario-b", "scenario-b__fxlms", 2),
        _window("scenario-b", "scenario-b__fxnlms", 3),
        _window("scenario-c", "scenario-c__fxlms", 4),
        _window("scenario-d", "scenario-d__fxlms", 5),
    ]
    splits = split_dataset(windows)
    assigned = {split: {window.scenario_id for window in values} for split, values in splits.items()}
    assert not assigned["train"] & assigned["validation"]
    assert not assigned["train"] & assigned["test"]
    normalization = compute_normalization(splits["train"])
    save_dataset(tmp_path, splits, normalization, WindowConfig(8, 4))
    reloaded, reloaded_normalization = load_dataset(tmp_path)
    assert sum(map(len, reloaded.values())) == len(windows)
    assert reloaded_normalization == normalization

