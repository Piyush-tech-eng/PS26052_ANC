from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anc.plant import ANCExperimentConfig, apply_causal_fir, run_anc_experiment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_ROOT / "experiments" / "m4_01_digital_anc_plant.py"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "m4_01_digital_anc_plant"


def test_digital_plant_obeys_documented_error_equation() -> None:
    reference = np.array([1.0, -0.5, 0.25, 0.0, 1.0])
    primary_path = np.array([0.0, 0.8, 0.2])
    secondary_path = np.array([0.0, 0.5, -0.1])
    result = run_anc_experiment(
        reference,
        primary_path,
        secondary_path,
        secondary_path,
        ANCExperimentConfig(4, 0.2, "fxnlms"),
    )

    np.testing.assert_allclose(
        result.primary_disturbance,
        apply_causal_fir(reference, primary_path),
    )
    np.testing.assert_allclose(
        result.secondary_path_output,
        apply_causal_fir(result.controller_output, secondary_path),
    )
    np.testing.assert_allclose(
        result.error,
        result.primary_disturbance + result.secondary_path_output,
    )


def test_filtered_x_reduces_error_against_no_control() -> None:
    rng = np.random.default_rng(9)
    reference = rng.normal(size=8_000)
    primary_path = np.array([0.0, 0.0, 0.85, 0.25, -0.10])
    secondary_path = np.array([0.0, 0.0, 0.55, 0.18, -0.08])
    baseline = run_anc_experiment(
        reference,
        primary_path,
        secondary_path,
        secondary_path,
        ANCExperimentConfig(24, 0.25, "none"),
    )
    controlled = run_anc_experiment(
        reference,
        primary_path,
        secondary_path,
        secondary_path,
        ANCExperimentConfig(24, 0.25, "fxnlms"),
    )

    assert controlled.final_error_power < baseline.final_error_power * 0.1


@pytest.fixture(scope="module", autouse=True)
def run_module_4_experiment() -> None:
    result = subprocess.run(
        [sys.executable, str(EXPERIMENT_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_module_4_end_to_end_acceptance_and_artifacts() -> None:
    summary = json.loads((RESULTS_DIRECTORY / "anc_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS"
    assert all(summary["acceptance"].values())

    required = [
        "reference_x.npy",
        "primary_disturbance_d.npy",
        "controller_output_y.npy",
        "secondary_path_output.npy",
        "error_e.npy",
        "controller_coefficients.npy",
        "primary_path.npy",
        "secondary_path_true.npy",
        "secondary_path_model_used.npy",
    ]
    for filename in required:
        assert np.isfinite(np.load(RESULTS_DIRECTORY / filename, allow_pickle=False)).all()
