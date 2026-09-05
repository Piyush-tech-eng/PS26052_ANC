from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_ROOT / "experiments" / "m3_07_end_to_end.py"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "m3_07_end_to_end"


@pytest.fixture(scope="module", autouse=True)
def run_module_3_experiment() -> None:
    result = subprocess.run(
        [sys.executable, str(EXPERIMENT_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_module_3_summary_passes_acceptance() -> None:
    summary = json.loads((RESULTS_DIRECTORY / "summary.json").read_text(encoding="utf-8"))
    assert summary["module"] == "M3"
    assert summary["status"] == "PASS"
    assert all(summary["acceptance"].values())


def test_module_3_required_artifacts_are_finite() -> None:
    required = [
        "adaptive_output.npy",
        "error.npy",
        "final_coefficients.npy",
        "coefficient_history.npy",
        "instantaneous_squared_error.npy",
        "mse_learning_curve.npy",
    ]
    for filename in required:
        values = np.load(RESULTS_DIRECTORY / filename, allow_pickle=False)
        assert values.size > 0
        assert np.isfinite(values).all()


def test_module_3_visual_diagnostics_exist() -> None:
    required = [
        "coefficient_trajectories.png",
        "instantaneous_squared_error.png",
        "mse_learning_curve.png",
        "lms_vs_nlms.png",
        "wiener_comparison.png",
    ]
    assert all((RESULTS_DIRECTORY / filename).is_file() for filename in required)
