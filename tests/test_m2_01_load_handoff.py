from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from anc.statistics import load_module1_handoff


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_DIRECTORY = PROJECT_ROOT / "results" / "m1_handoff"
SUMMARY_PATH = (
    PROJECT_ROOT
    / "results"
    / "m2_01_load_handoff"
    / "m2_input_summary.json"
)


def _write_handoff(
    directory: Path,
    *,
    reference: np.ndarray | None = None,
    desired: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    """Create a minimal temporary Module 1 handoff for validation tests."""

    directory.mkdir()

    np.save(
        directory / "reference.npy",
        np.array([1.0, -2.0, 3.0])
        if reference is None
        else reference,
    )
    np.save(
        directory / "desired.npy",
        np.array([0.5, -1.0, 2.0])
        if desired is None
        else desired,
    )

    if metadata is None:
        metadata = {
            "dataset": "module_1_to_module_2_handoff",
            "sampling_rate_hz": 16_000,
            "num_samples": 3,
        }

    (directory / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )


def test_load_module1_handoff_reads_canonical_inputs() -> None:
    """M2.01 reads the frozen canonical x[n], d[n], and metadata."""

    handoff = load_module1_handoff(HANDOFF_DIRECTORY)

    assert handoff.sampling_rate_hz == 16_000
    assert handoff.num_samples == 80_000
    assert handoff.reference.shape == (80_000,)
    assert handoff.desired.shape == (80_000,)
    assert handoff.reference.dtype == np.float64
    assert handoff.desired.dtype == np.float64
    assert np.isfinite(handoff.reference).all()
    assert np.isfinite(handoff.desired).all()
    assert handoff.metadata["num_samples"] == handoff.num_samples


def test_load_module1_handoff_rejects_unequal_signal_lengths(
    tmp_path: Path,
) -> None:
    handoff_directory = tmp_path / "m1_handoff"
    _write_handoff(
        handoff_directory,
        desired=np.array([0.5, -1.0]),
    )

    with pytest.raises(ValueError, match="equal lengths"):
        load_module1_handoff(handoff_directory)


def test_load_module1_handoff_rejects_nonfinite_samples(
    tmp_path: Path,
) -> None:
    handoff_directory = tmp_path / "m1_handoff"
    _write_handoff(
        handoff_directory,
        reference=np.array([1.0, np.nan, 3.0]),
    )

    with pytest.raises(ValueError, match="NaN or Inf"):
        load_module1_handoff(handoff_directory)


def test_load_module1_handoff_rejects_invalid_sampling_rate(
    tmp_path: Path,
) -> None:
    handoff_directory = tmp_path / "m1_handoff"
    _write_handoff(
        handoff_directory,
        metadata={
            "sampling_rate_hz": 0,
            "num_samples": 3,
        },
    )

    with pytest.raises(ValueError, match="sampling_rate_hz"):
        load_module1_handoff(handoff_directory)


def test_load_module1_handoff_rejects_metadata_count_mismatch(
    tmp_path: Path,
) -> None:
    handoff_directory = tmp_path / "m1_handoff"
    _write_handoff(
        handoff_directory,
        metadata={
            "sampling_rate_hz": 16_000,
            "num_samples": 99,
        },
    )

    with pytest.raises(ValueError, match="num_samples"):
        load_module1_handoff(handoff_directory)


def test_load_module1_handoff_requires_metadata_sample_count(
    tmp_path: Path,
) -> None:
    handoff_directory = tmp_path / "m1_handoff"
    _write_handoff(
        handoff_directory,
        metadata={
            "sampling_rate_hz": 16_000,
        },
    )

    with pytest.raises(ValueError, match="num_samples"):
        load_module1_handoff(handoff_directory)


def test_m2_01_experiment_writes_summary_without_changing_handoff() -> None:
    """The experiment writes only its M2 artifact, never M1 inputs."""

    original_handoff_files = {
        filename: (HANDOFF_DIRECTORY / filename).read_bytes()
        for filename in (
            "reference.npy",
            "desired.npy",
            "system_impulse_response.npy",
            "metadata.json",
        )
    }

    result = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "experiments" / "m2_01_load_handoff.py"),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"\nM2.01 experiment failed.\n"
        f"\nSTDOUT:\n{result.stdout}"
        f"\nSTDERR:\n{result.stderr}"
    )
    assert "M2.01" in result.stdout
    assert SUMMARY_PATH.exists()

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))

    assert summary["module"] == "M2.01"
    assert summary["sampling_rate_hz"] == 16_000
    assert summary["num_samples"] == 80_000
    assert summary["reference"]["symbol"] == "x[n]"
    assert summary["desired"]["symbol"] == "d[n]"
    assert all(summary["validation"].values())

    for filename, original_contents in original_handoff_files.items():
        assert (HANDOFF_DIRECTORY / filename).read_bytes() == original_contents
