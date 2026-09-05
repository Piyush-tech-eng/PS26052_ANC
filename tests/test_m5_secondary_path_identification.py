from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anc.plant import apply_causal_fir
from anc.secondary_path import (
    IdentificationConfig,
    SecondaryPathModel,
    identify_secondary_path,
    model_from_identification,
    validate_secondary_path_estimate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_PATH = PROJECT_ROOT / "experiments" / "m5_01_secondary_path_identification.py"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "m5_01_secondary_path_identification"


def test_identifier_uses_measured_data_and_converges(tmp_path: Path) -> None:
    rng = np.random.default_rng(11)
    true_path = np.array([0.0, 0.0, 0.6, -0.2, 0.08])
    excitation = rng.normal(size=7_000)
    measured_response = apply_causal_fir(excitation, true_path)
    result = identify_secondary_path(
        excitation,
        measured_response,
        IdentificationConfig(10, 0.35, "nlms"),
    )
    validation = validate_secondary_path_estimate(true_path, result.secondary_path_estimate)

    assert validation.relative_impulse_error < 1e-3
    assert np.mean(result.squared_error[-500:]) < np.mean(result.squared_error[:500])

    model = model_from_identification(result, validation)
    artifact = model.save(tmp_path / "secondary_path_model.npz")
    reloaded = SecondaryPathModel.load(artifact)
    np.testing.assert_allclose(reloaded.impulse_response, model.impulse_response)


@pytest.fixture(scope="module", autouse=True)
def run_module_5_experiment() -> None:
    result = subprocess.run(
        [sys.executable, str(EXPERIMENT_PATH)],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_module_5_end_to_end_identifies_and_exports_model() -> None:
    summary = json.loads(
        (RESULTS_DIRECTORY / "secondary_path_summary.json").read_text(encoding="utf-8")
    )
    assert summary["status"] == "PASS"
    assert all(summary["acceptance"].values())

    model = SecondaryPathModel.load(RESULTS_DIRECTORY / "secondary_path_model.npz")
    assert model.sampling_rate_hz == 8_000
    assert np.isfinite(model.impulse_response).all()
