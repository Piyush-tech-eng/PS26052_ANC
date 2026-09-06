from __future__ import annotations

import numpy as np

from anc.evaluation import build_scenario_matrix, run_scenario
from anc.plant import ANCExperimentConfig, apply_causal_fir
from anc.scenarios import ANCScenario, recording_from_array
from anc.secondary_path import SecondaryPathModel


def test_matrix_shares_scenario_id_between_controller_runs() -> None:
    definitions = build_scenario_matrix(signal_classes=("broadband",), input_levels=("nominal",), path_conditions=("nominal",),
                                        source_types=("synthetic",), controllers=("none", "fxnlms"), secondary_path_models=("identified",), seed=1)
    assert len(definitions) == 2
    assert definitions[0].scenario_id == definitions[1].scenario_id
    assert definitions[0].run_id != definitions[1].run_id


def test_run_scenario_reuses_replay_engine() -> None:
    definition = build_scenario_matrix(signal_classes=("broadband",), input_levels=("nominal",), path_conditions=("nominal",),
                                       source_types=("synthetic",), controllers=("none",), secondary_path_models=("identified",))[0]
    reference = np.random.default_rng(1).normal(size=512)
    model = SecondaryPathModel(np.array([0.0, 0.5, 0.2]), 8_000)
    scenario = ANCScenario(definition.scenario_id, recording_from_array(reference, sampling_rate_hz=8_000, source_type="synthetic"),
                           recording_from_array(apply_causal_fir(reference, [0.0, 0.8]), sampling_rate_hz=8_000, source_type="synthetic"), model, model.impulse_response)
    result = run_scenario(definition, scenario, ANCExperimentConfig(8, 0.05, "none", sampling_rate_hz=8_000))
    np.testing.assert_allclose(result.replay_result.residual, result.replay_result.measured_disturbance)
