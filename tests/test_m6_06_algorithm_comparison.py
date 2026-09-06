from __future__ import annotations

import json

import numpy as np

from anc.secondary_path import SecondaryPathModel
from m6_06_algorithm_comparison import run_algorithm_comparison


def test_m6_06_writes_complete_replay_artifact_set(tmp_path) -> None:
    model = SecondaryPathModel(np.array([0.0, 0.0, 0.62, 0.28, -0.10, 0.04]), 8_000)
    model_path = model.save(tmp_path / "model.npz")
    output = tmp_path / "m6"
    summary = run_algorithm_comparison(output, model_path, duration_seconds=0.25, create_plots=False)
    for name in ("scenario_manifest.json", "reference_x.npy", "error_input_or_disturbance.npy", "controller_output_y.npy",
                 "residual_error_e.npy", "filtered_reference.npy", "controller_coefficients.npy", "coefficient_history.npy",
                 "secondary_path_model_used.npz", "anc_replay_summary.json"):
        assert (output / name).is_file()
    assert set(summary["algorithms"]) == {"none", "fxlms", "fxnlms"}
    manifest = json.loads((output / "scenario_manifest.json").read_text())
    assert manifest["paths"]["true_secondary_path_available"] is True
