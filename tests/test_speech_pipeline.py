"""Tests for the speech/noise ingestion, mixing, and augmentation pipeline (Phase 2)."""

from __future__ import annotations

import numpy as np
import pytest

from anc.speech.sources import (
    NOISE_FAMILIES,
    NoiseSample,
    SpeechSample,
    generate_synthetic_impulsive,
    generate_synthetic_rotor,
    generate_synthetic_engine,
    generate_synthetic_wind,
    generate_synthetic_speech,
)
from anc.speech.mixing import MixedSample, mix_at_snr, mix_batch, DEFAULT_SNR_LADDER_DB
from anc.speech.augmentation import add_reverberation, apply_clipping, add_background_hum, apply_gain_variation
from anc.speech.resampling import resample_audio, normalize_audio_format


# ---------------------------------------------------------------------------
# SpeechSample / NoiseSample validation
# ---------------------------------------------------------------------------


class TestSpeechSample:

    def test_valid_construction(self) -> None:
        audio = np.random.randn(8000)
        sample = SpeechSample(audio=audio, sampling_rate=8000, source_id="test_01", speaker_id="spk_a")
        assert sample.num_samples == 8000
        assert sample.duration == pytest.approx(1.0)
        assert sample.speaker_id == "spk_a"

    def test_rejects_empty_audio(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            SpeechSample(audio=np.array([]), sampling_rate=8000, source_id="x", speaker_id="y")

    def test_rejects_nan(self) -> None:
        audio = np.array([1.0, float("nan"), 3.0])
        with pytest.raises(ValueError, match="NaN or Inf"):
            SpeechSample(audio=audio, sampling_rate=8000, source_id="x", speaker_id="y")

    def test_rejects_empty_speaker_id(self) -> None:
        with pytest.raises(ValueError, match="speaker_id"):
            SpeechSample(audio=np.ones(100), sampling_rate=8000, source_id="x", speaker_id="  ")

    def test_provenance_is_copied(self) -> None:
        prov = {"corpus": "test"}
        sample = SpeechSample(audio=np.ones(100), sampling_rate=8000, source_id="x", speaker_id="y", provenance=prov)
        prov["mutated"] = True
        assert "mutated" not in sample.provenance


class TestNoiseSample:

    def test_valid_construction(self) -> None:
        audio = np.random.randn(16000)
        sample = NoiseSample(audio=audio, sampling_rate=16000, noise_id="n01", noise_family="impulsive", source_id="gen")
        assert sample.num_samples == 16000
        assert sample.duration == pytest.approx(1.0)
        assert sample.noise_family == "impulsive"

    def test_rejects_invalid_family(self) -> None:
        with pytest.raises(ValueError, match="noise_family must be one of"):
            NoiseSample(audio=np.ones(100), sampling_rate=8000, noise_id="n", noise_family="invalid", source_id="s")

    def test_all_families_accepted(self) -> None:
        for family in NOISE_FAMILIES:
            sample = NoiseSample(audio=np.ones(100), sampling_rate=8000, noise_id="n", noise_family=family, source_id="s")
            assert sample.noise_family == family

    def test_is_synthetic_flag(self) -> None:
        sample = NoiseSample(
            audio=np.ones(100), sampling_rate=8000, noise_id="n",
            noise_family="broadband", source_id="s",
            provenance={"synthetic_approximation": True},
        )
        assert sample.is_synthetic is True


# ---------------------------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------------------------


class TestSyntheticGenerators:

    def test_impulsive_produces_finite_signal(self) -> None:
        sample = generate_synthetic_impulsive(8000, 1.0, seed=42)
        assert sample.noise_family == "impulsive"
        assert sample.num_samples == 8000
        assert np.isfinite(sample.audio).all()
        assert sample.is_synthetic

    def test_rotor_produces_finite_signal(self) -> None:
        sample = generate_synthetic_rotor(8000, 1.0, seed=42)
        assert sample.noise_family == "rotor"
        assert sample.num_samples == 8000
        assert np.isfinite(sample.audio).all()

    def test_engine_produces_finite_signal(self) -> None:
        sample = generate_synthetic_engine(8000, 1.0, seed=42)
        assert sample.noise_family == "engine_vehicle"
        assert np.isfinite(sample.audio).all()

    def test_wind_produces_finite_signal(self) -> None:
        sample = generate_synthetic_wind(8000, 1.0, seed=42)
        assert sample.noise_family == "wind"
        assert np.isfinite(sample.audio).all()

    def test_synthetic_speech_produces_finite_signal(self) -> None:
        sample = generate_synthetic_speech(8000, 1.0, seed=42)
        assert isinstance(sample, SpeechSample)
        assert sample.num_samples == 8000
        assert np.isfinite(sample.audio).all()

    def test_seed_reproducibility(self) -> None:
        a = generate_synthetic_impulsive(8000, 0.5, seed=123)
        b = generate_synthetic_impulsive(8000, 0.5, seed=123)
        np.testing.assert_array_equal(a.audio, b.audio)

    def test_different_seeds_different_output(self) -> None:
        a = generate_synthetic_impulsive(8000, 0.5, seed=1)
        b = generate_synthetic_impulsive(8000, 0.5, seed=2)
        assert not np.array_equal(a.audio, b.audio)


# ---------------------------------------------------------------------------
# Controlled-SNR mixing
# ---------------------------------------------------------------------------


class TestMixing:

    @staticmethod
    def _speech_and_noise(rate: int = 8000, length: int = 8000) -> tuple[SpeechSample, NoiseSample]:
        rng = np.random.default_rng(42)
        speech = SpeechSample(audio=rng.normal(0, 0.5, length), sampling_rate=rate, source_id="s1", speaker_id="spk1")
        noise = NoiseSample(audio=rng.normal(0, 0.3, length), sampling_rate=rate, noise_id="n1", noise_family="broadband", source_id="gen")
        return speech, noise

    def test_mix_produces_correct_equation(self) -> None:
        """m[n] = s[n] + alpha * v[n]"""
        speech, noise = self._speech_and_noise()
        mixed = mix_at_snr(speech, noise, target_snr_db=10.0)
        expected = mixed.clean_speech + mixed.noise_component
        np.testing.assert_allclose(mixed.noisy_speech, expected, atol=1e-12)

    def test_mix_preserves_clean_speech(self) -> None:
        speech, noise = self._speech_and_noise()
        mixed = mix_at_snr(speech, noise, target_snr_db=5.0)
        np.testing.assert_array_equal(mixed.clean_speech, speech.audio[:len(mixed.clean_speech)])

    def test_snr_is_approximately_correct(self) -> None:
        speech, noise = self._speech_and_noise(length=80000)  # Longer for stable power estimate
        for target_snr in [-5.0, 0.0, 5.0, 10.0, 15.0]:
            mixed = mix_at_snr(speech, noise, target_snr_db=target_snr)
            speech_power = np.mean(mixed.clean_speech ** 2)
            noise_power = np.mean(mixed.noise_component ** 2)
            actual_snr = 10 * np.log10(speech_power / max(noise_power, 1e-30))
            assert abs(actual_snr - target_snr) < 0.5, f"Target {target_snr}dB, got {actual_snr:.2f}dB"

    def test_provenance_metadata(self) -> None:
        speech, noise = self._speech_and_noise()
        mixed = mix_at_snr(speech, noise, target_snr_db=0.0)
        assert mixed.speech_source_id == "s1"
        assert mixed.speaker_id == "spk1"
        assert mixed.noise_source_id == "gen"
        assert mixed.noise_family == "broadband"
        assert mixed.snr_db == 0.0
        assert "computed_alpha" in mixed.provenance

    def test_mismatched_rates_rejected(self) -> None:
        speech = SpeechSample(audio=np.ones(100), sampling_rate=8000, source_id="s", speaker_id="sp")
        noise = NoiseSample(audio=np.ones(100), sampling_rate=16000, noise_id="n", noise_family="broadband", source_id="g")
        with pytest.raises(ValueError, match="Sampling rates must match"):
            mix_at_snr(speech, noise, 0.0)

    def test_trim_to_shorter(self) -> None:
        speech = SpeechSample(audio=np.ones(1000), sampling_rate=8000, source_id="s", speaker_id="sp")
        noise = NoiseSample(audio=np.ones(500), sampling_rate=8000, noise_id="n", noise_family="broadband", source_id="g")
        mixed = mix_at_snr(speech, noise, 0.0, trim_to_shorter=True)
        assert len(mixed.noisy_speech) == 500


class TestMixBatch:

    def test_cartesian_product(self) -> None:
        rng = np.random.default_rng(0)
        speeches = [SpeechSample(audio=rng.normal(0, 0.3, 800), sampling_rate=8000, source_id=f"s{i}", speaker_id=f"spk{i}") for i in range(2)]
        noises = [NoiseSample(audio=rng.normal(0, 0.2, 800), sampling_rate=8000, noise_id=f"n{i}", noise_family="broadband", source_id="g") for i in range(3)]
        results = mix_batch(speeches, noises, snr_ladder_db=(-5.0, 0.0, 5.0))
        # 2 speech × 3 noise × 3 SNR = 18
        assert len(results) == 18

    def test_empty_speech_rejected(self) -> None:
        noise = NoiseSample(audio=np.ones(100), sampling_rate=8000, noise_id="n", noise_family="broadband", source_id="g")
        with pytest.raises(ValueError, match="At least one speech"):
            mix_batch([], [noise])


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------


class TestAugmentation:

    def test_reverberation_preserves_length(self) -> None:
        audio = np.random.randn(8000)
        reverbed = add_reverberation(audio, 8000, rt60_seconds=0.3, seed=42)
        assert len(reverbed) == len(audio)
        assert np.isfinite(reverbed).all()

    def test_clipping_limits_amplitude(self) -> None:
        audio = np.array([0.0, 0.5, 1.0, -1.0, -0.5])
        clipped = apply_clipping(audio, threshold=0.6)
        assert np.max(np.abs(clipped)) <= 0.6 * np.max(np.abs(audio)) + 1e-10

    def test_clipping_rejects_zero_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold must be positive"):
            apply_clipping(np.ones(10), threshold=0.0)

    def test_background_hum_adds_tone(self) -> None:
        audio = np.zeros(8000)
        hummed = add_background_hum(audio, 8000, frequency_hz=50.0, amplitude=0.1)
        assert np.max(np.abs(hummed)) > 0  # Non-zero output

    def test_gain_variation(self) -> None:
        audio = np.ones(100) * 0.5
        gained = apply_gain_variation(audio, gain_db=6.0)
        expected_gain = 10 ** (6.0 / 20.0)
        np.testing.assert_allclose(gained, 0.5 * expected_gain, atol=1e-10)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


class TestResampling:

    def test_identity_resample(self) -> None:
        audio = np.random.randn(1000)
        result = resample_audio(audio, 8000, 8000)
        np.testing.assert_array_equal(result, audio)

    def test_downsample_reduces_length(self) -> None:
        audio = np.random.randn(16000)
        result = resample_audio(audio, 16000, 8000)
        assert len(result) == 8000

    def test_upsample_increases_length(self) -> None:
        audio = np.random.randn(8000)
        result = resample_audio(audio, 8000, 16000)
        assert len(result) == 16000

    def test_rejects_invalid_rates(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            resample_audio(np.ones(100), 0, 8000)

    def test_normalize_stereo_to_mono(self) -> None:
        stereo = np.random.randn(1000, 2)
        mono, rate = normalize_audio_format(stereo, 8000, 8000)
        assert mono.ndim == 1
        assert len(mono) == 1000
        assert rate == 8000

    def test_normalize_with_resample(self) -> None:
        audio = np.random.randn(16000)
        result, rate = normalize_audio_format(audio, 16000, 8000)
        assert rate == 8000
        assert len(result) == 8000

    def test_normalize_with_peak(self) -> None:
        audio = np.array([0.0, 0.5, -0.3, 0.8])
        result, _ = normalize_audio_format(audio, 8000, 8000, target_peak=1.0)
        assert np.max(np.abs(result)) == pytest.approx(1.0, abs=1e-10)
