"""Standardized classical-baseline package writers for Module 7."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from anc.evaluation.scenarios import ScenarioResult


@dataclass(frozen=True)
class BaselineEntry:
    scenario_id: str
    run_id: str
    algorithm: str
    metrics: dict[str, object]
    artifact_path: str


def _scalar_metrics(metrics: dict[str, object]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            output[key] = value
    return output


def build_benchmark_manifest(entries: Iterable[BaselineEntry]) -> dict[str, object]:
    records = [{"scenario_id": entry.scenario_id, "run_id": entry.run_id, "algorithm": entry.algorithm,
                "artifact_path": entry.artifact_path, "metrics": _scalar_metrics(entry.metrics)} for entry in entries]
    return {"format": "ps26052-anc-baseline-package", "format_version": 1, "runs": records}


def save_metrics_table(entries: Iterable[BaselineEntry], destination: str | Path) -> Path:
    destination = Path(destination)
    rows = [{"scenario_id": entry.scenario_id, "run_id": entry.run_id, "algorithm": entry.algorithm, **_scalar_metrics(entry.metrics)} for entry in entries]
    fields = sorted({key for row in rows for key in row})
    with destination.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return destination


def save_baseline_package(results: Iterable[ScenarioResult], output_dir: str | Path) -> tuple[list[BaselineEntry], Path]:
    """Persist controller outputs, per-run configs, standardized metrics, and a manifest."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    entries: list[BaselineEntry] = []
    for result in results:
        replay = result.replay_result
        relative_path = Path("baseline_outputs") / f"{result.run_id}.npz"
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, reference_x=replay.reference, baseline_residual=replay.residual,
                            controller_output_y=replay.controller_output, filtered_reference=replay.filtered_reference,
                            final_coefficients=replay.final_coefficients)
        config_dir = root / "scenario_config_snapshots"
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / f"{result.run_id}.json").write_text(json.dumps(result.definition.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        entries.append(BaselineEntry(result.definition.scenario_id, result.run_id, result.definition.controller, result.metrics, str(relative_path)))
    (root / "benchmark_manifest.json").write_text(json.dumps(build_benchmark_manifest(entries), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    save_metrics_table(entries, root / "metrics_table.csv")
    attenuation = [float(entry.metrics["attenuation_db_vs_no_control"]) for entry in entries if entry.metrics.get("attenuation_db_vs_no_control") is not None]
    summary = {"run_count": len(entries), "mean_attenuation_db_vs_no_control": None if not attenuation else float(np.mean(attenuation)),
               "divergent_runs": [entry.run_id for entry in entries if bool(getattr(entry.metrics.get("stability"), "get", lambda *_: False)("divergent", False))]}
    (root / "evaluation_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return entries, root
