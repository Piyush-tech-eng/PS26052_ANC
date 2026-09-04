from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

EXPERIMENT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "m2_10_final_integration.py"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "m2_10_final_integration"
)


def run_experiment() -> subprocess.CompletedProcess:
    """Run M2.10 from the project root."""

    return subprocess.run(
        [
            sys.executable,
            str(EXPERIMENT_PATH),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )


def test_m2_10_runs_successfully() -> None:

    result = run_experiment()

    assert result.returncode == 0, (
        "\nM2.10 failed.\n"
        f"\nSTDOUT:\n"
        f"{result.stdout}"
        f"\nSTDERR:\n"
        f"{result.stderr}"
    )


def test_m2_10_summary_exists_and_passes() -> None:

    summary_path = (
        RESULTS_DIRECTORY
        / "final_summary.json"
    )

    assert summary_path.is_file()

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert (
        summary["module"]
        == "M2.10"
    )

    assert (
        summary["status"]
        == "PASS"
    )


def test_m2_10_all_acceptance_checks_pass() -> None:

    summary_path = (
        RESULTS_DIRECTORY
        / "final_summary.json"
    )

    summary = json.loads(
        summary_path.read_text(
            encoding="utf-8"
        )
    )

    assert all(
        summary["acceptance"].values()
    )


def test_m2_10_core_outputs_exist() -> None:

    required_files = [
        "reference_x.npy",
        "desired_d.npy",
        "autocorrelation_rxx.npy",
        "cross_correlation_rxd.npy",
        "correlation_matrix_R.npy",
        "cross_correlation_vector_p.npy",
        "wiener_coefficients.npy",
        "estimated_desired_d_hat.npy",
        "error_e.npy",
        "true_impulse_response.npy",
        "estimated_impulse_response.npy",
        "coefficient_error.npy",
    ]

    for filename in required_files:

        path = (
            RESULTS_DIRECTORY
            / filename
        )

        assert path.is_file()


def test_m2_10_saved_arrays_are_finite() -> None:

    filenames = [
        "reference_x.npy",
        "desired_d.npy",
        "autocorrelation_rxx.npy",
        "cross_correlation_rxd.npy",
        "correlation_matrix_R.npy",
        "cross_correlation_vector_p.npy",
        "wiener_coefficients.npy",
        "estimated_desired_d_hat.npy",
        "error_e.npy",
    ]

    for filename in filenames:

        array = np.load(
            RESULTS_DIRECTORY
            / filename,
            allow_pickle=False,
        )

        assert np.isfinite(
            array
        ).all()


def test_m2_10_error_identity() -> None:

    desired = np.load(
        RESULTS_DIRECTORY
        / "desired_d.npy",
        allow_pickle=False,
    )

    estimate = np.load(
        RESULTS_DIRECTORY
        / "estimated_desired_d_hat.npy",
        allow_pickle=False,
    )

    error = np.load(
        RESULTS_DIRECTORY
        / "error_e.npy",
        allow_pickle=False,
    )

    assert np.allclose(
        error,
        desired - estimate,
        rtol=1e-12,
        atol=1e-12,
    )


def test_m2_10_wiener_equation() -> None:

    R = np.load(
        RESULTS_DIRECTORY
        / "correlation_matrix_R.npy",
        allow_pickle=False,
    )

    p = np.load(
        RESULTS_DIRECTORY
        / "cross_correlation_vector_p.npy",
        allow_pickle=False,
    )

    w_opt = np.load(
        RESULTS_DIRECTORY
        / "wiener_coefficients.npy",
        allow_pickle=False,
    )

    assert np.allclose(
        R @ w_opt,
        p,
        rtol=1e-8,
        atol=1e-10,
    )


def test_m2_10_correlation_matrix_is_symmetric() -> None:

    R = np.load(
        RESULTS_DIRECTORY
        / "correlation_matrix_R.npy",
        allow_pickle=False,
    )

    assert np.allclose(
        R,
        R.T,
        rtol=1e-10,
        atol=1e-12,
    )