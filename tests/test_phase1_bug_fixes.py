"""Tests for Phase 1 mandatory bug fixes: seed-pairing and split-leakage."""

from __future__ import annotations

import numpy as np
import pytest

from anc.evaluation.scenarios import ScenarioDefinition, build_scenario_matrix
from anc.evaluation.dataset import DatasetWindow, WindowConfig, extract_windows, split_dataset


# ---------------------------------------------------------------------------
# Fix 1: Seed-pairing — controllers share the same noise instance
# ---------------------------------------------------------------------------


class TestSeedPairing:
    """All controllers for the same scenario condition must receive the same seed."""

    def test_controllers_share_seed_within_scenario(self) -> None:
        """Verify none/fxlms/fxnlms for one condition get identical seeds."""
        definitions = build_scenario_matrix(
            signal_classes=("broadband",),
            input_levels=("nominal",),
            path_conditions=("nominal",),
            source_types=("synthetic",),
            controllers=("none", "fxlms", "fxnlms"),
            secondary_path_models=("identified",),
            seed=42,
        )
        # There should be 3 definitions: same scenario, 3 controllers
        assert len(definitions) == 3
        seeds = {d.seed for d in definitions}
        assert len(seeds) == 1, f"Expected 1 unique seed, got {seeds}"

    def test_different_conditions_get_different_seeds(self) -> None:
        """Different signal classes must produce different seeds."""
        definitions = build_scenario_matrix(
            signal_classes=("broadband", "tonal"),
            input_levels=("nominal",),
            path_conditions=("nominal",),
            source_types=("synthetic",),
            controllers=("none", "fxlms"),
            secondary_path_models=("identified",),
            seed=100,
        )
        # 2 conditions × 2 controllers = 4 definitions
        assert len(definitions) == 4

        # Group by scenario_id
        by_scenario: dict[str, list[ScenarioDefinition]] = {}
        for d in definitions:
            by_scenario.setdefault(d.scenario_id, []).append(d)

        scenario_seeds = [defs[0].seed for defs in by_scenario.values()]
        assert len(set(scenario_seeds)) == 2, "Different conditions must have different seeds"

        # Within each scenario, all controllers share the seed
        for scenario_id, defs in by_scenario.items():
            seeds = {d.seed for d in defs}
            assert len(seeds) == 1, f"Controllers in {scenario_id} should share a seed, got {seeds}"

    def test_multi_axis_seed_consistency(self) -> None:
        """Full Cartesian matrix: controllers never change the seed."""
        definitions = build_scenario_matrix(
            signal_classes=("broadband", "tonal"),
            input_levels=("low", "high"),
            path_conditions=("nominal",),
            source_types=("synthetic",),
            controllers=("none", "fxlms", "fxnlms"),
            secondary_path_models=("identified",),
            seed=0,
        )
        # 2 × 2 × 1 × 1 × 1 = 4 conditions, × 3 controllers = 12 definitions
        assert len(definitions) == 12

        by_scenario: dict[str, list[ScenarioDefinition]] = {}
        for d in definitions:
            by_scenario.setdefault(d.scenario_id, []).append(d)

        for scenario_id, defs in by_scenario.items():
            seeds = {d.seed for d in defs}
            assert len(seeds) == 1, f"Seed leak in {scenario_id}: {seeds}"

    def test_none_seed_stays_none(self) -> None:
        """When seed=None, all definitions must have seed=None."""
        definitions = build_scenario_matrix(
            signal_classes=("broadband",),
            input_levels=("nominal",),
            path_conditions=("nominal",),
            source_types=("synthetic",),
            controllers=("none", "fxlms"),
            secondary_path_models=("identified",),
            seed=None,
        )
        assert all(d.seed is None for d in definitions)


# ---------------------------------------------------------------------------
# Fix 2: Split-leakage — scenario grouping, not run_id grouping
# ---------------------------------------------------------------------------


class TestSplitLeakage:
    """Controller variants of the same scenario must stay in the same split."""

    @staticmethod
    def _make_windows(n_scenarios: int = 6, n_controllers: int = 3) -> list[DatasetWindow]:
        """Create synthetic windows spanning multiple scenarios × controllers."""
        windows = []
        for s in range(n_scenarios):
            scenario_id = f"scenario_{s:02d}"
            for c in range(n_controllers):
                controller = f"ctrl_{c}"
                run_id = f"{scenario_id}__{controller}"
                for w in range(4):  # 4 windows per run
                    windows.append(DatasetWindow(
                        reference_x=np.random.randn(512),
                        baseline_residual=np.random.randn(512),
                        controller_output_y=np.random.randn(512),
                        filtered_reference=np.random.randn(512),
                        target=None,
                        sample_rate=8000,
                        scenario_id=scenario_id,
                        run_id=run_id,
                        window_id=f"{run_id}__w{w:05d}",
                        algorithm_metadata={"controller": controller},
                        path_metadata={"path_condition": "nominal"},
                        source_type="synthetic",
                    ))
        return windows

    def test_no_scenario_leaks_across_splits(self) -> None:
        """Each scenario_id must appear in exactly one split."""
        windows = self._make_windows(n_scenarios=8, n_controllers=3)
        splits = split_dataset(windows)

        # Collect scenario_ids per split
        split_scenarios: dict[str, set[str]] = {}
        for split_name, split_windows in splits.items():
            split_scenarios[split_name] = {w.scenario_id for w in split_windows}

        # No scenario should appear in more than one split
        all_pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
        for a, b in all_pairs:
            overlap = split_scenarios[a] & split_scenarios[b]
            assert len(overlap) == 0, f"Scenario leakage between {a} and {b}: {overlap}"

    def test_all_controllers_in_same_split(self) -> None:
        """All controller variants of a scenario must be in the same split."""
        windows = self._make_windows(n_scenarios=6, n_controllers=3)
        splits = split_dataset(windows)

        # Build scenario→split mapping
        scenario_split: dict[str, str] = {}
        for split_name, split_windows in splits.items():
            for w in split_windows:
                if w.scenario_id in scenario_split:
                    assert scenario_split[w.scenario_id] == split_name, (
                        f"Scenario {w.scenario_id} found in both {scenario_split[w.scenario_id]} and {split_name}"
                    )
                else:
                    scenario_split[w.scenario_id] = split_name

    def test_all_windows_preserved(self) -> None:
        """Total window count must be preserved after splitting."""
        windows = self._make_windows(n_scenarios=6, n_controllers=2)
        splits = split_dataset(windows)
        total = sum(len(ws) for ws in splits.values())
        assert total == len(windows)

    def test_split_has_three_partitions(self) -> None:
        """Output must contain train, validation, and test."""
        windows = self._make_windows(n_scenarios=6, n_controllers=2)
        splits = split_dataset(windows)
        assert set(splits.keys()) == {"train", "validation", "test"}
        for name, ws in splits.items():
            assert len(ws) > 0, f"Split '{name}' is empty"
