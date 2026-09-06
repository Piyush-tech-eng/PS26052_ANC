from __future__ import annotations

import numpy as np
import pytest

from anc.io import (
    prepare_replay_inputs,
    recording_from_array,
)

from anc.plant.digital import (
    ANCExperimentConfig,
    apply_causal_fir,
)

from anc.replay import (
    run_anc_replay,
)


def make_replay_inputs(
    *,
    num_samples: int = 8_000,
):

    rng = np.random.default_rng(
        26052
    )

    reference_samples = rng.normal(
        0.0,
        0.25,
        size=num_samples,
    )

    primary_path = np.array(
        [
            0.0,
            0.0,
            0.8,
            0.3,
            -0.1,
        ],
        dtype=np.float64,
    )

    measured_samples = (
        apply_causal_fir(
            reference_samples,
            primary_path,
        )
    )

    reference = recording_from_array(
        reference_samples,
        sampling_rate_hz=8_000,
        role="reference",
    )

    measured = recording_from_array(
        measured_samples,
        sampling_rate_hz=8_000,
        role="measured",
    )

    replay_inputs = (
        prepare_replay_inputs(
            reference,
            measured,
            align=False,
        )
    )

    return (
        replay_inputs,
        primary_path,
    )


def make_paths():

    secondary_true = np.array(
        [
            0.0,
            0.0,
            0.65,
            0.25,
            -0.08,
        ],
        dtype=np.float64,
    )

    secondary_model = (
        secondary_true.copy()
    )

    return (
        secondary_true,
        secondary_model,
    )


def test_replay_output_lengths() -> None:

    replay_inputs, _ = (
        make_replay_inputs()
    )

    secondary_true, secondary_model = (
        make_paths()
    )

    config = ANCExperimentConfig(
        filter_length=16,
        step_size=0.05,
        algorithm="fxnlms",
        sampling_rate_hz=8_000,
    )

    result = run_anc_replay(
        replay_inputs,
        secondary_true,
        secondary_model,
        config,
    )

    num_samples = (
        replay_inputs.overlap_samples
    )

    assert len(
        result.reference
    ) == num_samples

    assert len(
        result.measured_disturbance
    ) == num_samples

    assert len(
        result.controller_output
    ) == num_samples

    assert len(
        result.secondary_path_output
    ) == num_samples

    assert len(
        result.residual
    ) == num_samples


def test_no_control_matches_measured_signal() -> None:

    replay_inputs, _ = (
        make_replay_inputs()
    )

    secondary_true, secondary_model = (
        make_paths()
    )

    config = ANCExperimentConfig(
        filter_length=16,
        step_size=0.05,
        algorithm="none",
        sampling_rate_hz=8_000,
    )

    result = run_anc_replay(
        replay_inputs,
        secondary_true,
        secondary_model,
        config,
    )

    np.testing.assert_allclose(
        result.controller_output,
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.secondary_path_output,
        0.0,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        result.residual,
        result.measured_disturbance,
        atol=1e-12,
    )


def test_fxlms_reduces_residual_power() -> None:

    replay_inputs, _ = (
        make_replay_inputs(
            num_samples=20_000
        )
    )

    secondary_true, secondary_model = (
        make_paths()
    )

    config = ANCExperimentConfig(
        filter_length=16,
        step_size=0.05,
        algorithm="fxnlms",
        sampling_rate_hz=8_000,
        learning_window=256,
    )

    result = run_anc_replay(
        replay_inputs,
        secondary_true,
        secondary_model,
        config,
    )

    assert np.isfinite(
        result.residual
    ).all()

    assert np.isfinite(
        result.final_coefficients
    ).all()

    assert (
        result.final_residual_power
        < result.initial_residual_power
    )


def test_sampling_rate_mismatch_rejected() -> None:

    replay_inputs, _ = (
        make_replay_inputs()
    )

    secondary_true, secondary_model = (
        make_paths()
    )

    config = ANCExperimentConfig(
        filter_length=16,
        step_size=0.05,
        algorithm="fxnlms",
        sampling_rate_hz=16_000,
    )

    with pytest.raises(
        ValueError,
        match="sampling_rate_hz",
    ):
        run_anc_replay(
            replay_inputs,
            secondary_true,
            secondary_model,
            config,
        )


def test_initial_coefficients_length_rejected() -> None:

    replay_inputs, _ = (
        make_replay_inputs()
    )

    secondary_true, secondary_model = (
        make_paths()
    )

    config = ANCExperimentConfig(
        filter_length=16,
        step_size=0.05,
        algorithm="fxnlms",
        sampling_rate_hz=8_000,
    )

    with pytest.raises(
        ValueError,
        match="initial_coefficients",
    ):
        run_anc_replay(
            replay_inputs,
            secondary_true,
            secondary_model,
            config,
            initial_coefficients=np.zeros(
                5
            ),
        )


def test_attenuation_property_is_finite() -> None:

    replay_inputs, _ = (
        make_replay_inputs(
            num_samples=20_000
        )
    )

    secondary_true, secondary_model = (
        make_paths()
    )

    config = ANCExperimentConfig(
        filter_length=16,
        step_size=0.05,
        algorithm="fxnlms",
        sampling_rate_hz=8_000,
    )

    result = run_anc_replay(
        replay_inputs,
        secondary_true,
        secondary_model,
        config,
    )

    assert np.isfinite(
        result.attenuation_db
    )