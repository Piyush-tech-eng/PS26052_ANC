"""Dataset-level (deterministic, saved) augmentation for speech and noise.

These augmentations are applied at dataset-generation time and become part of
the frozen, versioned dataset — ensuring reproducible results. Training-time
random augmentation (Section 3.6 of the AI plan) is kept separate in the AI
training pipeline so that Module 7 stays framework-agnostic.
"""

from __future__ import annotations

import numpy as np


def add_reverberation(
    audio: np.ndarray,
    sampling_rate: int,
    *,
    rt60_seconds: float = 0.3,
    seed: int | None = None,
) -> np.ndarray:
    """Apply synthetic reverberation using an exponentially decaying impulse response.

    Parameters
    ----------
    audio : np.ndarray
        Input mono audio signal.
    sampling_rate : int
        Sampling rate in Hz.
    rt60_seconds : float
        Approximate RT60 reverberation time.
    seed : int, optional
        Random seed for reproducible impulse response generation.

    Returns
    -------
    np.ndarray
        Reverberated audio, same length as input.
    """
    rng = np.random.default_rng(seed)
    ir_length = int(rt60_seconds * sampling_rate)
    if ir_length <= 0:
        return audio.copy()

    # Generate random impulse response with exponential decay
    decay = np.exp(-np.linspace(0, 6.9, ir_length))  # ~60dB decay over RT60
    ir = rng.normal(0, 1, ir_length) * decay
    ir[0] = 1.0  # Direct path
    ir = ir / np.sqrt(np.sum(ir ** 2))  # Normalize energy

    # Convolve and trim to original length
    reverbed = np.convolve(audio, ir, mode="full")[:len(audio)]
    return reverbed.astype(np.float64)


def apply_clipping(
    audio: np.ndarray,
    *,
    threshold: float = 0.8,
) -> np.ndarray:
    """Apply hard clipping to simulate distorted/overdriven input.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    threshold : float
        Clipping level relative to the signal's peak amplitude.
        Values above ``threshold * peak`` are clipped.

    Returns
    -------
    np.ndarray
        Clipped audio.
    """
    if threshold <= 0:
        raise ValueError("threshold must be positive.")

    peak = np.max(np.abs(audio))
    if peak < np.finfo(np.float64).eps:
        return audio.copy()

    clip_level = threshold * peak
    return np.clip(audio, -clip_level, clip_level).astype(np.float64)


def add_background_hum(
    audio: np.ndarray,
    sampling_rate: int,
    *,
    frequency_hz: float = 50.0,
    amplitude: float = 0.02,
) -> np.ndarray:
    """Add a low-frequency sinusoidal hum (e.g., 50/60 Hz mains interference).

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    sampling_rate : int
        Sampling rate in Hz.
    frequency_hz : float
        Hum frequency (typically 50 or 60 Hz).
    amplitude : float
        Hum amplitude relative to signal.

    Returns
    -------
    np.ndarray
        Audio with added hum.
    """
    t = np.arange(len(audio)) / sampling_rate
    hum = amplitude * np.sin(2 * np.pi * frequency_hz * t)
    return (audio + hum).astype(np.float64)


def apply_gain_variation(
    audio: np.ndarray,
    *,
    gain_db: float = 0.0,
) -> np.ndarray:
    """Apply a fixed gain in dB to simulate level variations.

    Parameters
    ----------
    audio : np.ndarray
        Input audio signal.
    gain_db : float
        Gain to apply in dB.

    Returns
    -------
    np.ndarray
        Gain-adjusted audio.
    """
    gain_linear = 10.0 ** (gain_db / 20.0)
    return (audio * gain_linear).astype(np.float64)


# ---------------------------------------------------------------------------
# Real room impulse response (RIR) augmentation
# ---------------------------------------------------------------------------

def load_rir_corpus(
    rir_dir: str | Path,
    *,
    target_sr: int = 16_000,
    max_rirs: int | None = None,
) -> list[np.ndarray]:
    """Load real room impulse responses from a directory.

    Parameters
    ----------
    rir_dir : str or Path
        Directory containing ``.wav`` RIR files.
    target_sr : int
        Target sample rate.  RIRs are resampled if needed.
    max_rirs : int, optional
        Maximum number of RIRs to load.

    Returns
    -------
    list of np.ndarray
        Each element is a 1-D float64 impulse response.
    """
    from pathlib import Path

    rir_path = Path(rir_dir)
    if not rir_path.exists():
        return []

    rirs: list[np.ndarray] = []
    extensions = {".wav", ".flac"}

    for audio_file in sorted(rir_path.rglob("*")):
        if audio_file.suffix.lower() not in extensions:
            continue
        if max_rirs is not None and len(rirs) >= max_rirs:
            break

        try:
            # Try soundfile first, fall back to scipy
            try:
                import soundfile as sf
                ir, sr = sf.read(str(audio_file), dtype="float64", always_2d=False)
            except ImportError:
                from scipy.io import wavfile
                sr, ir = wavfile.read(str(audio_file))
                if ir.dtype in (np.int16, np.int32):
                    ir = ir.astype(np.float64) / np.iinfo(ir.dtype).max
                ir = ir.astype(np.float64)

            # Mono downmix
            if ir.ndim == 2:
                ir = np.mean(ir, axis=1)

            # Resample if needed
            if sr != target_sr:
                from anc.speech.resampling import resample_audio
                ir = resample_audio(ir, sr, target_sr)

            # Normalize energy
            ir_energy = np.sqrt(np.sum(ir ** 2))
            if ir_energy > np.finfo(np.float64).eps:
                ir = ir / ir_energy

            rirs.append(ir)
        except Exception:
            continue

    return rirs


def convolve_with_rir(
    audio: np.ndarray,
    rir: np.ndarray,
) -> np.ndarray:
    """Apply reverberation by convolving audio with a real room impulse response.

    Parameters
    ----------
    audio : np.ndarray
        Input mono audio signal.
    rir : np.ndarray
        Room impulse response (1-D float64).

    Returns
    -------
    np.ndarray
        Reverberated audio, same length as input.
    """
    if rir.size == 0:
        return audio.copy()

    reverbed = np.convolve(audio, rir, mode="full")[:len(audio)]
    return reverbed.astype(np.float64)


def generate_synthetic_rirs(
    n_rirs: int = 10,
    sampling_rate: int = 16_000,
    *,
    rt60_range: tuple[float, float] = (0.2, 0.8),
    seed: int | None = None,
) -> list[np.ndarray]:
    """Generate synthetic RIRs using exponentially decaying noise.

    Useful as a fallback when no real RIR corpus is available.

    Parameters
    ----------
    n_rirs : int
        Number of RIRs to generate.
    sampling_rate : int
        Sampling rate in Hz.
    rt60_range : tuple of float
        Range of RT60 values (in seconds) to sample from.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    list of np.ndarray
        Synthetic impulse responses.
    """
    rng = np.random.default_rng(seed)
    rirs: list[np.ndarray] = []

    for _ in range(n_rirs):
        rt60 = rng.uniform(rt60_range[0], rt60_range[1])
        ir_length = int(rt60 * sampling_rate)
        if ir_length <= 0:
            continue

        # Exponentially decaying noise (approximates diffuse field)
        decay = np.exp(-np.linspace(0, 6.9, ir_length))  # ~60dB decay over RT60
        ir = rng.normal(0, 1, ir_length) * decay
        ir[0] = 1.0  # Direct path

        # Normalize energy
        ir_energy = np.sqrt(np.sum(ir ** 2))
        if ir_energy > np.finfo(np.float64).eps:
            ir = ir / ir_energy

        rirs.append(ir.astype(np.float64))

    return rirs
