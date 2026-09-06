from __future__ import annotations

import json

from m7_01_scenario_matrix_evaluation import build_evaluation_package


def test_baseline_package_contains_standard_artifacts(tmp_path) -> None:
    build_evaluation_package(tmp_path, sample_count=1_024, create_plots=False)
    for name in ("benchmark_manifest.json", "metrics_table.csv", "evaluation_summary.json", "dataset_manifest.json", "normalization.json"):
        assert (tmp_path / name).is_file()
    manifest = json.loads((tmp_path / "benchmark_manifest.json").read_text())
    assert len(manifest["runs"]) == 24
    assert all((tmp_path / record["artifact_path"]).is_file() for record in manifest["runs"])
