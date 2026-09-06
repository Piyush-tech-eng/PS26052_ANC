from __future__ import annotations

import numpy as np
import pytest
from scipy.io import wavfile

from anc.io import (
    load_npy_recording,
    load_wav_recording,
    prepare_replay_inputs,
    recording_from_array,
)


def test_recording_from_mono_array() -> None:

    samples = np.array(
        [0.1, -0.2, 0.3],
        dtype=np.float32,
    )

    recording = recording_from_array(
        samples,
        sampling_rate_hz=8_000,
        role="reference",
    )

    assert (
        recording.num_samples
        == 3
    )

    assert (
        recording.selected_channel
        == 0
    )

    assert (
        recording.total_channels
        == 1
    )

    assert (
        recording.sampling_rate_hz
        == 8_000
    )


def test_multichannel_selection() -> None:

    samples = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
        ],
        dtype=np.float64,
    )

    recording = recording_from_array(
        samples,
        sampling_rate_hz=8_000,
        role="measured",
        channel=1,
    )

    np.testing.assert_allclose(
        recording.samples,
        np.array(
            [10.0, 20.0, 30.0]
        ),
    )


def test_load_npy_recording(
    tmp_path,
) -> None:

    path = (
        tmp_path
        / "signal.npy"
    )

    expected = np.array(
        [0.1, 0.2, 0.3],
        dtype=np.float64,
    )

    np.save(
        path,
        expected,
    )

    recording = (
        load_npy_recording(
            path,
            sampling_rate_hz=16_000,
            role="reference",
        )
    )

    np.testing.assert_allclose(
        recording.samples,
        expected,
    )

    assert (
        recording.sampling_rate_hz
        == 16_000
    )


def test_load_wav_recording(
    tmp_path,
) -> None:

    path = (
        tmp_path
        / "signal.wav"
    )

    samples = np.array(
        [
            0,
            1000,
            -1000,
            2000,
        ],
        dtype=np.int16,
    )

    wavfile.write(
        path,
        8_000,
        samples,
    )

    recording = (
        load_wav_recording(
            path,
            role="measured",
        )
    )

    assert (
        recording.sampling_rate_hz
        == 8_000
    )

    assert (
        recording.num_samples
        == len(samples)
    )

    assert np.isfinite(
        recording.samples
    ).all()


def test_prepare_replay_inputs_with_alignment() -> None:

    rng = np.random.default_rng(
        26052
    )

    reference_samples = rng.normal(
        size=2_000
    )

    delay = 80

    measured_samples = np.zeros_like(
        reference_samples
    )

    measured_samples[
        delay:
    ] = reference_samples[
        :-delay
    ]

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

    replay = prepare_replay_inputs(
        reference,
        measured,
        align=True,
        max_delay_samples=200,
        minimum_overlap_samples=500,
    )

    assert (
        replay.delay_samples
        == delay
    )

    np.testing.assert_allclose(
        replay.reference.samples,
        replay.measured.samples,
        atol=1e-12,
    )


def test_sampling_rate_mismatch_rejected() -> None:

    reference = recording_from_array(
        np.ones(100),
        sampling_rate_hz=8_000,
        role="reference",
    )

    measured = recording_from_array(
        np.ones(100),
        sampling_rate_hz=16_000,
        role="measured",
    )

    with pytest.raises(
        ValueError,
        match="Sampling-rate mismatch",
    ):
        prepare_replay_inputs(
            reference,
            measured,
        )


def test_nan_is_rejected() -> None:

    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        recording_from_array(
            np.array(
                [1.0, np.nan]
            ),
            sampling_rate_hz=8_000,
            role="reference",
        )