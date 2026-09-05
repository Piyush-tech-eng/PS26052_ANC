import numpy as np
import pytest

from anc.adaptive import (
    run_lms_step_size,
    run_lms_step_size_sweep,
)


def test_step_size_run_returns_stable_result() -> None:

    reference = np.ones(
        500,
        dtype=np.float64,
    )

    desired = 2.0 * reference

    result = run_lms_step_size(
        reference,
        desired,
        filter_length=1,
        step_size=0.01,
    )

    assert result.stable

    assert result.mse is not None

    assert result.final_error_power is not None

    assert result.initial_error_power is not None


def test_step_size_run_detects_convergence() -> None:

    reference = np.ones(
        500,
        dtype=np.float64,
    )

    desired = 2.0 * reference

    result = run_lms_step_size(
        reference,
        desired,
        filter_length=1,
        step_size=0.01,
    )

    assert result.stable

    assert result.converged


def test_invalid_step_size_is_unstable() -> None:

    result = run_lms_step_size(
        np.ones(10),
        np.ones(10),
        filter_length=1,
        step_size=-0.1,
    )

    assert not result.stable

    assert not result.converged

    assert result.reason is not None


def test_step_size_sweep_returns_one_result_per_mu() -> None:

    reference = np.ones(
        100,
        dtype=np.float64,
    )

    desired = 2.0 * reference

    step_sizes = [
        0.001,
        0.01,
        0.05,
    ]

    results = run_lms_step_size_sweep(
        reference,
        desired,
        filter_length=1,
        step_sizes=step_sizes,
    )

    assert len(
        results
    ) == len(
        step_sizes
    )

    returned_step_sizes = [
        result.step_size
        for result in results
    ]

    assert returned_step_sizes == (
        step_sizes
    )


def test_empty_step_size_sweep_rejected() -> None:

    with pytest.raises(
        ValueError
    ):

        run_lms_step_size_sweep(
            np.ones(10),
            np.ones(10),
            filter_length=1,
            step_sizes=[],
        )