from __future__ import annotations

from anc.evaluation import load_dataset
from m7_01_scenario_matrix_evaluation import build_evaluation_package


def test_module_7_acceptance_builds_loadable_leakage_aware_package(tmp_path) -> None:
    summary = build_evaluation_package(tmp_path, sample_count=1_024, create_plots=False)
    splits, normalization = load_dataset(tmp_path)
    assert summary["run_count"] == 24
    assert all(splits[name] for name in ("train", "validation", "test"))
    assert normalization["policy"].startswith("z-score")
    run_sets = [{window.run_id for window in split} for split in splits.values()]
    assert not run_sets[0] & run_sets[1] & run_sets[2]
