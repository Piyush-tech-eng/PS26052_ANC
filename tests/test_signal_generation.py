import numpy as np
import pytest

from anc.signals.generators import (
    generate_sine,
    generate_multi_tone,
    generate_white_noise,
    generate_colored_noise,
    generate_impulse,
)


def test_sine_number_of_samples():
    signal = generate_sine(
        sampling_rate=16000,
        duration=2.0,
        frequency=440.0,
        )

    assert signal.num_samples == 32000

def test_sine_sampling_rate():
    signal = generate_sine(
        sampling_rate=16000,
        duration=2.0,
        frequency=440.0,
    )

    assert signal.sampling_rate == 16000


def test_sine_duration():
    signal = generate_sine(
        sampling_rate=16000,
        duration=2.0,
        frequency=440.0,
    )

    assert np.isclose(signal.duration, 2.0)


def test_sine_amplitude():
    signal = generate_sine(
        sampling_rate=16000,
        duration=1.0,
        frequency=440.0,
        amplitude=2.0,
    )

    assert np.max(signal.samples) <= 2.0
    assert np.min(signal.samples) >= -2.0

def test_multi_tone_number_of_samples():
    signal = generate_multi_tone(
        sampling_rate=16000,
        duration=2.0,
        frequencies=[200.0,440.0, 1000.0],
        amplitudes=[1.0, 0.7, 0.4],
    )

    assert signal.num_samples == 32000

def test_multi_tone_sampling_rate():
    signal = generate_multi_tone(
        sampling_rate=16000,
        duration=1.0,
        frequencies=[200.0,440.0],
        amplitudes=[1.0, 0.5],
    )

    assert signal.sampling_rate == 16000

def test_multi_tone_metadata():
    frequencies = [200.0, 440.0, 1000.0]
    amplitudes = [1.0, 0.7, 0.4]

    signal = generate_multi_tone(
        sampling_rate=16000,
        duration=1.0,
        frequencies=frequencies,
        amplitudes=amplitudes,
    )

    assert signal.metadata["signal_type"] == "multi_tone"

    assert (
        signal.metadata
    )

def test_white_noise_number_of_samples():
    signal = generate_white_noise(
        sampling_rate=16000,
        duration=2.0,
        std=1.0,
        seed=42,
    )

    assert signal.num_samples == 32000

def test_white_noise_number_of_samples():
    signal = generate_white_noise(
        sampling_rate=16000,
        duration=2.0,
        std=1.0,
        seed=42,
    )

    assert signal.num_samples == 32000

def test_white_noise_reproducibility():
    signal_a = generate_white_noise(
        sampling_rate=16000,
        duration=1.0,
        std=1.0,
        seed=42,
    )

    signal_b = generate_white_noise(
        sampling_rate=16000,
        duration=1.0,
        std=1.0,
        seed=42,
    )

    assert np.array_equal(
        signal_a.samples,
        signal_b.samples,
    )

def test_colored_noise_number_of_samples():
    signal = generate_colored_noise(
        sampling_rate=16000,
        duration=2.0,
        std=1.0,
        alpha=0.95,
        seed=42,
    )

    assert signal.num_samples == 32000

def test_colored_noise_reproducibility():
    signal_a = generate_colored_noise(
        sampling_rate=16000,
        duration=1.0,
        alpha=0.95,
        seed=42,
    )

    signal_b = generate_colored_noise(
        sampling_rate=16000,
        duration=1.0,
        alpha=0.95,
        seed=42,
    )

    assert np.array_equal(
        signal_a.samples,
        signal_b.samples,
    )

def test_generate_impulse() -> None:
    signal = generate_impulse(
        sampling_rate=1000,
        duration=0.01,
        index=4,
        amplitude=2.5,
    )

    expected = np.zeros(10)
    expected[4] = 2.5

    np.testing.assert_array_equal(
        signal.samples,
        expected,
    )

def test_generate_impulse_rejects_invalid_index() -> None:
    with pytest.raises(ValueError):
        generate_impulse(
            sampling_rate=1000,
            duration=0.01,
            index=10,
        )