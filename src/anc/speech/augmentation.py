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
