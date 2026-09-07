"""Source-identity data objects for speech and noise samples.

These dataclasses provide provenance-bearing representations for clean speech
and noise audio, matching the PS26052 problem statement's requirements for
curated defence noise categories and multi-speaker speech corpora.

The ``noise_family`` taxonomy maps directly to the PS statement:
    impulsive (gunshot/artillery), rotor (helicopter/drone), engine_vehicle,
    broadband, colored, wind, alarm_siren.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# Noise family taxonomy matching the PS26052 problem statement
NOISE_FAMILIES = frozenset({
    "impulsive",       # gunshot, artillery
    "rotor",           # helicopter, drone
    "engine_vehicle",  # vehicle engines
    "broadband",       # white/pink noise
    "colored",         # colored noise
    "wind",            # wind noise
    "alarm_siren",     # alarms, sirens
    "mixed",           # synthetic composite
})


def _finite_mono_vector(values: np.ndarray | list[float], *, name: str) -> np.ndarray:
    """Validate and return a finite one-dimensional float64 vector."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or Inf.")
    return array.copy()


@dataclass(frozen=True)
class SpeechSample:
    """One clean speech sample with provenance metadata.

    Attributes
    ----------
    audio : np.ndarray
        Mono float64 waveform, peak-normalized or raw.
    sampling_rate : int
        Sampling frequency in Hz.
    source_id : str
        Unique identifier for this audio clip (e.g., file path hash).
    speaker_id : str
        Speaker identity for split-by-speaker grouping.
    duration : float
        Duration in seconds (computed from audio length).
    provenance : dict
        License, corpus name, original path, etc.
    """

    audio: np.ndarray
    sampling_rate: int
    source_id: str
    speaker_id: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "audio",
            _finite_mono_vector(self.audio, name="audio"),
        )
        if not isinstance(self.sampling_rate, int) or self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be a positive integer.")
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string.")
        if not isinstance(self.speaker_id, str) or not self.speaker_id.strip():
            raise ValueError("speaker_id must be a non-empty string.")
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def duration(self) -> float:
        """Signal duration in seconds."""
        return float(len(self.audio) / self.sampling_rate)

    @property
    def num_samples(self) -> int:
        """Number of audio samples."""
        return len(self.audio)


@dataclass(frozen=True)
class NoiseSample:
    """One noise sample with family classification and provenance.

    Attributes
    ----------
    audio : np.ndarray
        Mono float64 waveform.
    sampling_rate : int
        Sampling frequency in Hz.
    noise_id : str
        Unique identifier for this noise clip.
    noise_family : str
        Category from the PS26052 taxonomy (e.g., 'impulsive', 'rotor').
    source_id : str
        Provenance source identifier.
    provenance : dict
        License, origin, synthetic_approximation flag, etc.
    """

    audio: np.ndarray
    sampling_rate: int
    noise_id: str
    noise_family: str
    source_id: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "audio",
            _finite_mono_vector(self.audio, name="audio"),
        )
        if not isinstance(self.sampling_rate, int) or self.sampling_rate <= 0:
            raise ValueError("sampling_rate must be a positive integer.")
        if not isinstance(self.noise_id, str) or not self.noise_id.strip():
            raise ValueError("noise_id must be a non-empty string.")
        if self.noise_family not in NOISE_FAMILIES:
            supported = ", ".join(sorted(NOISE_FAMILIES))
            raise ValueError(
                f"noise_family must be one of: {supported}. "
                f"Got: '{self.noise_family}'."
            )
        if not isinstance(self.source_id, str) or not self.source_id.strip():
            raise ValueError("source_id must be a non-empty string.")
        object.__setattr__(self, "provenance", dict(self.provenance))

    @property
    def duration(self) -> float:
        """Signal duration in seconds."""
        return float(len(self.audio) / self.sampling_rate)

    @property
    def num_samples(self) -> int:
        """Number of audio samples."""
        return len(self.audio)

    @property
    def is_synthetic(self) -> bool:
        """Whether this noise was synthetically generated."""
        return bool(self.provenance.get("synthetic_approximation", False))


# ---------------------------------------------------------------------------
# Synthetic noise generators (defence-specific approximations)
# ---------------------------------------------------------------------------


def generate_synthetic_impulsive(
    sampling_rate: int,
    duration: float,
    *,
    num_impulses: int = 5,
    seed: int | None = None,
) -> NoiseSample:
    """Generate synthetic impulsive noise approximating gunshot/artillery transients.

    Uses random exponentially-decaying bursts to mimic impulsive defence sounds.
    Tagged with ``synthetic_approximation=True`` in provenance.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(sampling_rate * duration)
    audio = np.zeros(num_samples, dtype=np.float64)

    for _ in range(num_impulses):
        position = rng.integers(0, max(1, num_samples - int(0.01 * sampling_rate)))
        burst_len = min(int(rng.uniform(0.002, 0.015) * sampling_rate), num_samples - position)
        amplitude = rng.uniform(0.5, 1.0)
        decay = np.exp(-np.linspace(0, 8, burst_len))
        noise_burst = amplitude * decay * rng.normal(0, 1, burst_len)
        audio[position:position + burst_len] += noise_burst

    return NoiseSample(
        audio=audio, sampling_rate=sampling_rate,
        noise_id=f"synth_impulsive_seed{seed}", noise_family="impulsive",
        source_id="synthetic_generator",
        provenance={"synthetic_approximation": True, "generator": "impulsive_burst", "seed": seed, "num_impulses": num_impulses},
    )


def generate_synthetic_rotor(
    sampling_rate: int,
    duration: float,
    *,
    fundamental_hz: float = 18.0,
    num_harmonics: int = 12,
    seed: int | None = None,
) -> NoiseSample:
    """Generate synthetic rotor noise approximating helicopter/drone sounds.

    Uses harmonic series with amplitude roll-off plus broadband turbulence.
    Tagged with ``synthetic_approximation=True`` in provenance.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(sampling_rate * duration)
    t = np.arange(num_samples) / sampling_rate
    audio = np.zeros(num_samples, dtype=np.float64)

    # Harmonic tonal components (rotor blade-pass frequency + harmonics)
    for h in range(1, num_harmonics + 1):
        freq = fundamental_hz * h
        if freq >= sampling_rate / 2:
            break
        amplitude = 0.3 / (h ** 0.7)  # Amplitude roll-off
        phase = rng.uniform(0, 2 * np.pi)
        audio += amplitude * np.sin(2 * np.pi * freq * t + phase)

    # Broadband turbulence component
    turbulence = rng.normal(0, 0.08, num_samples)
    # Low-pass the turbulence for realism
    alpha = 0.92
    for i in range(1, num_samples):
        turbulence[i] = alpha * turbulence[i - 1] + (1 - alpha) * turbulence[i]
    audio += turbulence

    return NoiseSample(
        audio=audio, sampling_rate=sampling_rate,
        noise_id=f"synth_rotor_seed{seed}", noise_family="rotor",
        source_id="synthetic_generator",
        provenance={"synthetic_approximation": True, "generator": "harmonic_rotor", "seed": seed,
                     "fundamental_hz": fundamental_hz, "num_harmonics": num_harmonics},
    )


def generate_synthetic_engine(
    sampling_rate: int,
    duration: float,
    *,
    rpm_hz: float = 45.0,
    seed: int | None = None,
) -> NoiseSample:
    """Generate synthetic engine/vehicle noise.

    Low-frequency harmonic drone with broadband mechanical noise.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(sampling_rate * duration)
    t = np.arange(num_samples) / sampling_rate
    audio = np.zeros(num_samples, dtype=np.float64)

    # Engine fundamental + harmonics
    for h in range(1, 8):
        freq = rpm_hz * h
        if freq >= sampling_rate / 2:
            break
        amplitude = 0.25 / (h ** 0.5)
        audio += amplitude * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))

    # Mechanical broadband noise
    audio += 0.1 * rng.normal(0, 1, num_samples)

    return NoiseSample(
        audio=audio, sampling_rate=sampling_rate,
        noise_id=f"synth_engine_seed{seed}", noise_family="engine_vehicle",
        source_id="synthetic_generator",
        provenance={"synthetic_approximation": True, "generator": "engine_harmonic", "seed": seed, "rpm_hz": rpm_hz},
    )


def generate_synthetic_wind(
    sampling_rate: int,
    duration: float,
    *,
    seed: int | None = None,
) -> NoiseSample:
    """Generate synthetic wind noise using heavily filtered broadband noise."""
    rng = np.random.default_rng(seed)
    num_samples = int(sampling_rate * duration)

    # Start with white noise
    audio = rng.normal(0, 0.3, num_samples)

    # Heavy low-pass filtering for wind character (multiple passes)
    alpha = 0.97
    for _ in range(3):
        for i in range(1, num_samples):
            audio[i] = alpha * audio[i - 1] + (1 - alpha) * audio[i]

    # Add slow amplitude modulation (gusts)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * np.arange(num_samples) / sampling_rate)
    audio *= modulation

    return NoiseSample(
        audio=audio, sampling_rate=sampling_rate,
        noise_id=f"synth_wind_seed{seed}", noise_family="wind",
        source_id="synthetic_generator",
        provenance={"synthetic_approximation": True, "generator": "filtered_broadband_wind", "seed": seed},
    )


def generate_synthetic_speech(
    sampling_rate: int,
    duration: float,
    *,
    speaker_id: str = "synth_speaker_0",
    seed: int | None = None,
) -> SpeechSample:
    """Generate a synthetic speech-like signal for pipeline testing.

    Uses amplitude-modulated filtered noise to approximate speech envelope
    characteristics. Not for actual model training — only for pipeline
    validation and integration testing.
    """
    rng = np.random.default_rng(seed)
    num_samples = int(sampling_rate * duration)

    # Formant-like filtered noise
    audio = rng.normal(0, 0.2, num_samples)

    # Band-pass effect: mild low-pass then high-pass
    alpha_lp = 0.85
    for i in range(1, num_samples):
        audio[i] = alpha_lp * audio[i - 1] + (1 - alpha_lp) * audio[i]

    # Speech-like amplitude modulation (syllable rate ~4 Hz)
    t = np.arange(num_samples) / sampling_rate
    envelope = np.maximum(0, np.sin(2 * np.pi * 4.0 * t + rng.uniform(0, 2 * np.pi)))
    envelope = np.power(envelope, 0.5)  # Soften edges
    audio *= envelope

    # Add some formant-like tonal content
    for freq in [300.0, 800.0, 2500.0]:
        if freq < sampling_rate / 2:
            amplitude = rng.uniform(0.02, 0.06)
            audio += amplitude * envelope * np.sin(2 * np.pi * freq * t + rng.uniform(0, 2 * np.pi))

    return SpeechSample(
        audio=audio, sampling_rate=sampling_rate,
        source_id=f"synth_speech_seed{seed}", speaker_id=speaker_id,
        provenance={"synthetic_approximation": True, "generator": "formant_modulated_noise", "seed": seed},
    )
