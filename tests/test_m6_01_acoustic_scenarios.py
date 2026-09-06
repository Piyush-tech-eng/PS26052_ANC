from __future__ import annotations

import wave

import numpy as np
import pytest

from anc.scenarios import (
    ANCScenario,
    AcousticRecording,
    load_npy_recording,
    load_wav_recording,
    recording_from_array,
)
from anc.secondary_path import SecondaryPathModel


def make_secondary_path_model(
    *,
    sampling_rate_hz: int = 8_000,
) -> SecondaryPathModel:
    return SecondaryPathModel(
        impulse_response=np.array(
            [0.0, 0.55, 0.18, -0.08],
            dtype=np.float64,
        ),
        sampling_rate_hz=sampling_rate_hz,
        metadata={
            "test": True,
        },
    )


def test_recording_from_array_preserves_signal_contract() -> None:
    samples = np.array(
        [0.25, -0.5, 1.0, 0.0],
        dtype=np.float64,
    )

    recording = recording_from_array(
        samples,
        sampling_rate_hz=8_000,
        source_type="synthetic",
        provenance={
            "generator": "unit-test",
        },
    )

    np.testing.assert_allclose(
        recording.samples,
        samples,
    )

    assert recording.num_samples == 4
    assert recording.sampling_rate_hz == 8_000
    assert recording.source_type == "synthetic"

    assert recording.duration_seconds == pytest.approx(
        4 / 8_000,
    )


def test_recording_rejects_nonfinite_samples() -> None:
    with pytest.raises(
        ValueError,
        match="NaN or Inf",
    ):
        AcousticRecording(
            samples=np.array(
                [0.0, np.nan],
            ),
            sampling_rate_hz=8_000,
            source_type="synthetic",
        )


def test_npy_recording_loader(tmp_path) -> None:
    path = tmp_path / "recording.npy"

    samples = np.array(
        [0.1, -0.2, 0.3],
        dtype=np.float64,
    )

    np.save(
        path,
        samples,
    )

    recording = load_npy_recording(
        path,
        sampling_rate_hz=16_000,
    )

    np.testing.assert_allclose(
        recording.samples,
        samples,
    )

    assert recording.source_type == "recorded"
    assert recording.sampling_rate_hz == 16_000
    assert recording.provenance["loader"] == "npy"


def test_wav_loader_selects_requested_channel(
    tmp_path,
) -> None:
    path = tmp_path / "stereo.wav"

    channel_zero = np.array(
        [0, 1_000, -1_000, 2_000],
        dtype=np.int16,
    )

    channel_one = np.array(
        [500, -500, 1_500, -1_500],
        dtype=np.int16,
    )

    interleaved = np.column_stack(
        (
            channel_zero,
            channel_one,
        )
    )

    with wave.open(
        str(path),
        "wb",
    ) as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8_000)

        wav_file.writeframes(
            interleaved.astype(
                "<i2",
            ).tobytes()
        )

    recording = load_wav_recording(
        path,
        channel=1,
    )

    expected = (
        channel_one.astype(
            np.float64,
        )
        / float(2**15)
    )

    np.testing.assert_allclose(
        recording.samples,
        expected,
    )

    assert recording.sampling_rate_hz == 8_000
    assert recording.channel_metadata[
        "original_channels"
    ] == 2

    assert recording.channel_metadata[
        "selected_channel"
    ] == 1


def test_scenario_requires_matching_sample_rates() -> None:
    model = make_secondary_path_model(
        sampling_rate_hz=8_000,
    )

    reference = recording_from_array(
        np.ones(16),
        sampling_rate_hz=8_000,
        source_type="synthetic",
    )

    error_input = recording_from_array(
        np.ones(16),
        sampling_rate_hz=16_000,
        source_type="synthetic",
    )

    with pytest.raises(
        ValueError,
        match="same sampling rate",
    ):
        ANCScenario(
            scenario_id="invalid-rate",
            reference=reference,
            error_input=error_input,
            secondary_path_model=model,
        )


def test_scenario_requires_matching_lengths() -> None:
    model = make_secondary_path_model()

    reference = recording_from_array(
        np.ones(16),
        sampling_rate_hz=8_000,
        source_type="synthetic",
    )

    error_input = recording_from_array(
        np.ones(15),
        sampling_rate_hz=8_000,
        source_type="synthetic",
    )

    with pytest.raises(
        ValueError,
        match="same number of samples",
    ):
        ANCScenario(
            scenario_id="invalid-length",
            reference=reference,
            error_input=error_input,
            secondary_path_model=model,
        )


def test_scenario_requires_model_rate_match() -> None:
    model = make_secondary_path_model(
        sampling_rate_hz=16_000,
    )

    reference = recording_from_array(
        np.ones(16),
        sampling_rate_hz=8_000,
        source_type="synthetic",
    )

    error_input = recording_from_array(
        np.ones(16),
        sampling_rate_hz=8_000,
        source_type="synthetic",
    )

    with pytest.raises(
        ValueError,
        match="sampling rate",
    ):
        ANCScenario(
            scenario_id="invalid-model-rate",
            reference=reference,
            error_input=error_input,
            secondary_path_model=model,
        )


def test_valid_scenario_exposes_common_properties() -> None:
    model = make_secondary_path_model()

    reference = recording_from_array(
        np.ones(32),
        sampling_rate_hz=8_000,
        source_type="synthetic",
    )

    error_input = recording_from_array(
        np.zeros(32),
        sampling_rate_hz=8_000,
        source_type="recorded",
    )

    scenario = ANCScenario(
        scenario_id="m6-valid-scenario",
        reference=reference,
        error_input=error_input,
        secondary_path_model=model,
        metadata={
            "purpose": "unit-test",
        },
    )

    assert scenario.sampling_rate_hz == 8_000
    assert scenario.num_samples == 32

    assert scenario.source_types == {
        "reference": "synthetic",
        "error_input": "recorded",
    }