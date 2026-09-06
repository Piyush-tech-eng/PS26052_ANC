"""Scenario definitions and execution bridge for the Module 7 benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
import json
from typing import Any, Iterable

import numpy as np

from anc.io import prepare_replay_inputs, recording_from_array
from anc.plant import ANCExperimentConfig
from anc.replay import ANCReplayResult, run_anc_replay
from anc.scenarios import ANCScenario
from anc.evaluation.metrics import evaluate_scenario


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value.lower()).strip("-")


@dataclass(frozen=True)
class ScenarioDefinition:
    """Data-only description of one condition and one controller run."""

    scenario_id: str
    signal_class: str
    input_level: str
    path_condition: str
    source_type: str
    controller: str
    secondary_path_model: str
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("scenario_id", "signal_class", "input_level", "path_condition", "source_type", "controller", "secondary_path_model"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string.")
        if self.source_type not in {"synthetic", "recorded"}:
            raise ValueError("source_type must be 'synthetic' or 'recorded'.")
        if self.seed is not None and (not isinstance(self.seed, int) or isinstance(self.seed, bool)):
            raise TypeError("seed must be an integer or None.")
        object.__setattr__(self, "config_snapshot", dict(self.config_snapshot))
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def run_id(self) -> str:
        return f"{self.scenario_id}__{_safe_id(self.controller)}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id, "run_id": self.run_id, "signal_class": self.signal_class,
            "input_level": self.input_level, "path_condition": self.path_condition, "source_type": self.source_type,
            "controller": self.controller, "secondary_path_model": self.secondary_path_model,
            "config_snapshot": self.config_snapshot, "seed": self.seed, "provenance": self.provenance,
        }


def build_scenario_matrix(
    *, signal_classes: Iterable[str], input_levels: Iterable[str], path_conditions: Iterable[str],
    source_types: Iterable[str], controllers: Iterable[str], secondary_path_models: Iterable[str],
    config_snapshot: dict[str, Any] | None = None, seed: int | None = None,
) -> list[ScenarioDefinition]:
    """Create a reproducible Cartesian scenario matrix; controllers share condition IDs."""

    axes = [list(axis) for axis in (signal_classes, input_levels, path_conditions, source_types, secondary_path_models, controllers)]
    if any(not axis for axis in axes):
        raise ValueError("Every scenario axis must contain at least one value.")
    definitions: list[ScenarioDefinition] = []
    for index, (signal_class, input_level, path_condition, source_type, model, controller) in enumerate(product(*axes)):
        scenario_id = "__".join((_safe_id(str(signal_class)), _safe_id(str(input_level)), _safe_id(str(path_condition)), _safe_id(str(source_type)), _safe_id(str(model))))
        definitions.append(ScenarioDefinition(
            scenario_id=scenario_id, signal_class=str(signal_class), input_level=str(input_level),
            path_condition=str(path_condition), source_type=str(source_type), controller=str(controller),
            secondary_path_model=str(model), config_snapshot=dict(config_snapshot or {}),
            seed=None if seed is None else seed + index,
        ))
    return definitions


@dataclass(frozen=True)
class ScenarioResult:
    """One replay result plus standard metrics and its provenance-bearing definition."""

    definition: ScenarioDefinition
    scenario: ANCScenario
    replay_result: ANCReplayResult
    metrics: dict[str, object]
    target: np.ndarray | None = None

    @property
    def run_id(self) -> str:
        return self.definition.run_id


def run_scenario(
    definition: ScenarioDefinition, scenario: ANCScenario, config: ANCExperimentConfig,
    *, target: np.ndarray | None = None, baseline_residual: np.ndarray | None = None,
) -> ScenarioResult:
    """Run an ANCScenario through the existing replay engine without controller duplication."""

    if not isinstance(definition, ScenarioDefinition) or not isinstance(scenario, ANCScenario):
        raise TypeError("definition and scenario must be ScenarioDefinition and ANCScenario instances.")
    if config.algorithm != definition.controller:
        raise ValueError("config.algorithm must match definition.controller.")
    if scenario.true_secondary_path is None:
        raise ValueError("Replay evaluation requires a measured secondary path; unavailable physical paths are not fabricated.")
    reference = recording_from_array(scenario.reference.samples, sampling_rate_hz=scenario.sampling_rate_hz, role="reference", source=scenario.reference.source_type)
    measured = recording_from_array(scenario.error_input.samples, sampling_rate_hz=scenario.sampling_rate_hz, role="measured", source=scenario.error_input.source_type)
    replay_inputs = prepare_replay_inputs(reference, measured, align=False)
    replay = run_anc_replay(replay_inputs, scenario.true_secondary_path, scenario.secondary_path_model.impulse_response, config)
    target_array = None if target is None else np.asarray(target, dtype=np.float64).copy()
    return ScenarioResult(
        definition=definition, scenario=scenario, replay_result=replay,
        metrics=evaluate_scenario(replay.residual, sampling_rate_hz=scenario.sampling_rate_hz,
                                  baseline_residual=baseline_residual, target=target_array,
                                  coefficient_history=replay.coefficient_history, convergence_window=config.learning_window),
        target=target_array,
    )
