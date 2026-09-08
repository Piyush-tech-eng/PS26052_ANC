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


# ---------------------------------------------------------------------------
# Real corpus loaders
# ---------------------------------------------------------------------------

# Mapping from ESC-50 category names to PS26052 noise families
_ESC50_FAMILY_MAP: dict[str, str] = {
    "siren": "alarm_siren",
    "engine": "engine_vehicle",
    "car_horn": "alarm_siren",
    "dog": "broadband",
    "rain": "broadband",
    "sea_waves": "broadband",
    "wind": "wind",
    "thunderstorm": "impulsive",
    "fireworks": "impulsive",
    "gunshot": "impulsive",
    "hand_saw": "engine_vehicle",
    "chainsaw": "engine_vehicle",
    "helicopter": "rotor",
    "airplane": "engine_vehicle",
    "crackling_fire": "broadband",
    "clock_alarm": "alarm_siren",
    "train": "engine_vehicle",
}


def _read_audio_file(
    path: Path,
    target_sr: int = 16_000,
) -> tuple[np.ndarray, int]:
    """Read an audio file and resample to target sample rate.

    Tries ``soundfile`` first, falls back to ``scipy.io.wavfile``.
    Returns mono float64 audio and the target sample rate.
    """
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path), dtype="float64", always_2d=False)
    except ImportError:
        from scipy.io import wavfile
        sr, audio = wavfile.read(str(path))
        # Normalize integer PCM to float64
        if audio.dtype in (np.int16, np.int32):
            audio = audio.astype(np.float64) / np.iinfo(audio.dtype).max
        elif audio.dtype == np.uint8:
            audio = (audio.astype(np.float64) - 128) / 128.0
        audio = audio.astype(np.float64)

    # Downmix to mono if multi-channel
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)

    # Resample if needed
    if sr != target_sr:
        from anc.speech.resampling import resample_audio
        audio = resample_audio(audio, sr, target_sr)

    return audio, target_sr


def load_librispeech(
    root_dir: str | Path,
    *,
    target_sr: int = 16_000,
    max_samples: int | None = None,
) -> list[SpeechSample]:
    """Load LibriSpeech corpus as SpeechSample objects.

    Parameters
    ----------
    root_dir : str or Path
        Path to the LibriSpeech root (e.g. ``data/corpora/librispeech``).
        Should contain ``LibriSpeech/train-clean-100/`` etc.
    target_sr : int
        Target sample rate for all output.
    max_samples : int, optional
        Limit the number of samples loaded (for testing/debugging).

    Returns
    -------
    list of SpeechSample
    """
    root = Path(root_dir)
    samples: list[SpeechSample] = []
    count = 0

    # LibriSpeech structure: LibriSpeech/{subset}/{speaker_id}/{chapter_id}/{file}.flac
    for flac_path in sorted(root.rglob("*.flac")):
        if max_samples is not None and count >= max_samples:
            break

        parts = flac_path.relative_to(root).parts
        # Expect at least: LibriSpeech / subset / speaker / chapter / file.flac
        if len(parts) < 4:
            continue

        speaker_id = f"ls_{parts[-3]}"
        source_id = f"librispeech_{flac_path.stem}"

        try:
            audio, sr = _read_audio_file(flac_path, target_sr)
        except Exception:
            continue

        if audio.size == 0:
            continue

        samples.append(SpeechSample(
            audio=audio,
            sampling_rate=sr,
            source_id=source_id,
            speaker_id=speaker_id,
            provenance={
                "dataset": "librispeech",
                "license": "CC-BY-4.0",
                "original_path": str(flac_path.relative_to(root)),
                "synthetic_approximation": False,
            },
        ))
        count += 1

    return samples


def load_vctk(
    root_dir: str | Path,
    *,
    target_sr: int = 16_000,
    max_samples: int | None = None,
) -> list[SpeechSample]:
    """Load VCTK corpus as SpeechSample objects.

    Parameters
    ----------
    root_dir : str or Path
        Path to the VCTK root (e.g. ``data/corpora/vctk``).
        Should contain ``VCTK-Corpus-0.92/wav48_silence_trimmed/``.
    target_sr : int
        Target sample rate for all output.
    max_samples : int, optional
        Limit the number of samples loaded.

    Returns
    -------
    list of SpeechSample
    """
    root = Path(root_dir)
    samples: list[SpeechSample] = []
    count = 0

    # VCTK structure: VCTK-Corpus-0.92/wav48_silence_trimmed/{speaker_id}/{file}.flac
    wav_root = root
    for candidate in [
        root / "VCTK-Corpus-0.92" / "wav48_silence_trimmed",
        root / "wav48_silence_trimmed",
    ]:
        if candidate.exists():
            wav_root = candidate
            break

    for audio_path in sorted(wav_root.rglob("*.flac")):
        if max_samples is not None and count >= max_samples:
            break

        parts = audio_path.relative_to(wav_root).parts
        if len(parts) < 2:
            continue

        speaker_id = f"vctk_{parts[0]}"
        source_id = f"vctk_{audio_path.stem}"

        try:
            audio, sr = _read_audio_file(audio_path, target_sr)
        except Exception:
            continue

        if audio.size == 0:
            continue

        samples.append(SpeechSample(
            audio=audio,
            sampling_rate=sr,
            source_id=source_id,
            speaker_id=speaker_id,
            provenance={
                "dataset": "vctk",
                "license": "CC-BY-4.0",
                "original_path": str(audio_path.relative_to(root)),
                "synthetic_approximation": False,
            },
        ))
        count += 1

    # Also try .wav files (some VCTK versions have wav instead of flac)
    if not samples:
        for audio_path in sorted(wav_root.rglob("*.wav")):
            if max_samples is not None and count >= max_samples:
                break

            parts = audio_path.relative_to(wav_root).parts
            if len(parts) < 2:
                continue

            speaker_id = f"vctk_{parts[0]}"
            source_id = f"vctk_{audio_path.stem}"

            try:
                audio, sr = _read_audio_file(audio_path, target_sr)
            except Exception:
                continue

            if audio.size == 0:
                continue

            samples.append(SpeechSample(
                audio=audio,
                sampling_rate=sr,
                source_id=source_id,
                speaker_id=speaker_id,
                provenance={
                    "dataset": "vctk",
                    "license": "CC-BY-4.0",
                    "original_path": str(audio_path.relative_to(root)),
                    "synthetic_approximation": False,
                },
            ))
            count += 1

    return samples


def load_noise_directory(
    root_dir: str | Path,
    family: str,
    *,
    dataset_name: str = "custom",
    license_id: str = "unknown",
    target_sr: int = 16_000,
    max_samples: int | None = None,
) -> list[NoiseSample]:
    """Load noise audio files from a directory.

    Parameters
    ----------
    root_dir : str or Path
        Directory containing ``.wav`` or ``.flac`` noise files.
    family : str
        Noise family to assign (must be in ``NOISE_FAMILIES``).
    dataset_name : str
        Name of the source dataset for provenance.
    license_id : str
        License identifier for provenance.
    target_sr : int
        Target sample rate.
    max_samples : int, optional
        Limit the number of samples loaded.

    Returns
    -------
    list of NoiseSample
    """
    if family not in NOISE_FAMILIES:
        supported = ", ".join(sorted(NOISE_FAMILIES))
        raise ValueError(f"noise_family must be one of: {supported}. Got: '{family}'.")

    root = Path(root_dir)
    samples: list[NoiseSample] = []
    count = 0

    extensions = {".wav", ".flac", ".ogg", ".mp3"}
    audio_files = sorted(
        f for f in root.rglob("*") if f.suffix.lower() in extensions
    )

    for audio_path in audio_files:
        if max_samples is not None and count >= max_samples:
            break

        noise_id = f"{dataset_name}_{family}_{audio_path.stem}"
        source_id = f"{dataset_name}_{audio_path.stem}"

        try:
            audio, sr = _read_audio_file(audio_path, target_sr)
        except Exception:
            continue

        if audio.size == 0:
            continue

        samples.append(NoiseSample(
            audio=audio,
            sampling_rate=sr,
            noise_id=noise_id,
            noise_family=family,
            source_id=source_id,
            provenance={
                "dataset": dataset_name,
                "license": license_id,
                "original_path": str(audio_path.relative_to(root)),
                "synthetic_approximation": False,
            },
        ))
        count += 1

    return samples


def load_musan_noise(
    root_dir: str | Path,
    *,
    target_sr: int = 16_000,
    max_per_family: int | None = None,
) -> list[NoiseSample]:
    """Load MUSAN noise subset with family classification.

    MUSAN's noise directory is mapped to PS26052 families:
    - free-sound → broadband (default mapping, refined by filename)
    - sound-bible → impulsive / broadband

    Parameters
    ----------
    root_dir : str or Path
        MUSAN root (containing ``musan/noise/``).
    target_sr : int
        Target sample rate.
    max_per_family : int, optional
        Max samples per noise family.

    Returns
    -------
    list of NoiseSample
    """
    root = Path(root_dir)

    # Find the noise directory
    noise_dir = None
    for candidate in [root / "musan" / "noise", root / "noise"]:
        if candidate.exists():
            noise_dir = candidate
            break

    if noise_dir is None:
        return []

    family_counts: dict[str, int] = {}
    samples: list[NoiseSample] = []

    for audio_path in sorted(noise_dir.rglob("*.wav")):
        # Simple heuristic family assignment based on path/name
        name_lower = audio_path.stem.lower()
        if any(kw in name_lower for kw in ["gun", "shot", "explo", "bang"]):
            family = "impulsive"
        elif any(kw in name_lower for kw in ["wind", "breeze"]):
            family = "wind"
        elif any(kw in name_lower for kw in ["engine", "car", "vehicle", "truck", "motor"]):
            family = "engine_vehicle"
        elif any(kw in name_lower for kw in ["siren", "alarm", "horn"]):
            family = "alarm_siren"
        elif any(kw in name_lower for kw in ["helicopter", "chopper", "drone", "rotor"]):
            family = "rotor"
        else:
            family = "broadband"

        if max_per_family is not None:
            if family_counts.get(family, 0) >= max_per_family:
                continue

        try:
            audio, sr = _read_audio_file(audio_path, target_sr)
        except Exception:
            continue

        if audio.size == 0:
            continue

        samples.append(NoiseSample(
            audio=audio,
            sampling_rate=sr,
            noise_id=f"musan_{family}_{audio_path.stem}",
            noise_family=family,
            source_id=f"musan_{audio_path.stem}",
            provenance={
                "dataset": "musan",
                "license": "Various (see MUSAN documentation)",
                "original_path": str(audio_path.relative_to(root)),
                "synthetic_approximation": False,
            },
        ))
        family_counts[family] = family_counts.get(family, 0) + 1

    return samples


def load_esc50_noise(
    root_dir: str | Path,
    *,
    target_sr: int = 16_000,
    max_per_family: int | None = None,
) -> list[NoiseSample]:
    """Load ESC-50 dataset with PS26052 noise family mapping.

    Parameters
    ----------
    root_dir : str or Path
        ESC-50 root (containing ``ESC-50-master/audio/``).
    target_sr : int
        Target sample rate.
    max_per_family : int, optional
        Max samples per noise family.

    Returns
    -------
    list of NoiseSample
    """
    import csv

    root = Path(root_dir)

    # Find audio and metadata
    audio_dir = None
    meta_path = None
    for prefix in [root / "ESC-50-master", root]:
        if (prefix / "audio").exists():
            audio_dir = prefix / "audio"
            if (prefix / "meta" / "esc50.csv").exists():
                meta_path = prefix / "meta" / "esc50.csv"
            break

    if audio_dir is None:
        return []

    # Load category mapping from CSV if available
    file_to_category: dict[str, str] = {}
    if meta_path is not None and meta_path.exists():
        with open(meta_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "")
                category = row.get("category", "").lower().replace(" ", "_")
                if filename and category:
                    file_to_category[filename] = category

    family_counts: dict[str, int] = {}
    samples: list[NoiseSample] = []

    for audio_path in sorted(audio_dir.rglob("*.wav")):
        category = file_to_category.get(audio_path.name, "")
        family = _ESC50_FAMILY_MAP.get(category)

        if family is None:
            # Skip categories that don't map to our taxonomy
            continue

        if max_per_family is not None:
            if family_counts.get(family, 0) >= max_per_family:
                continue

        try:
            audio, sr = _read_audio_file(audio_path, target_sr)
        except Exception:
            continue

        if audio.size == 0:
            continue

        samples.append(NoiseSample(
            audio=audio,
            sampling_rate=sr,
            noise_id=f"esc50_{family}_{audio_path.stem}",
            noise_family=family,
            source_id=f"esc50_{audio_path.stem}",
            provenance={
                "dataset": "esc50",
                "license": "CC-BY-NC-3.0",
                "category": category,
                "original_path": str(audio_path.relative_to(root)),
                "synthetic_approximation": False,
            },
        ))
        family_counts[family] = family_counts.get(family, 0) + 1

    return samples
