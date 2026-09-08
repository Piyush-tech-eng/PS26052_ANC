"""Sample-rate and format normalization utilities.

Ensures any audio input — regardless of original sample rate, channel count,
or bit depth — is converted to the model's canonical format (mono, float64,
target sample rate) before entering the pipeline. This prevents the system
from only working on inputs pre-shaped to match training data.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import resample_poly
from math import gcd


def resample_audio(
    audio: np.ndarray,
    original_rate: int,
    target_rate: int,
) -> np.ndarray:
    """Resample a mono audio signal to a target sample rate.

    Uses polyphase filtering (scipy.signal.resample_poly) for high-quality
    integer-ratio resampling.

    Parameters
    ----------
    audio : np.ndarray
        Input mono audio (1-D float64).
    original_rate : int
        Original sampling rate in Hz.
    target_rate : int
        Target sampling rate in Hz.

    Returns
    -------
    np.ndarray
        Resampled audio at the target rate.
    """
    if original_rate <= 0 or target_rate <= 0:
        raise ValueError("Both original_rate and target_rate must be positive.")

    if original_rate == target_rate:
        return audio.copy()

    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 1:
        raise ValueError("audio must be one-dimensional.")

    # Compute up/down factors
    common = gcd(original_rate, target_rate)
    up = target_rate // common
    down = original_rate // common

    resampled = resample_poly(audio, up, down)
    return np.asarray(resampled, dtype=np.float64)


def normalize_audio_format(
    audio: np.ndarray,
    original_rate: int,
    target_rate: int,
    *,
    target_peak: float | None = None,
) -> tuple[np.ndarray, int]:
    """Convert audio to canonical format: mono, float64, target sample rate.

    Parameters
    ----------
    audio : np.ndarray
        Input audio, can be 1-D (mono) or 2-D (multi-channel, shape: [samples, channels]).
    original_rate : int
        Original sampling rate.
    target_rate : int
        Desired output sampling rate.
    target_peak : float, optional
        If provided, peak-normalize the output to this level.

    Returns
    -------
    tuple of (np.ndarray, int)
        (normalized_audio, target_rate)
    """
    audio = np.asarray(audio, dtype=np.float64)

    # Convert to mono if multi-channel
    if audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    elif audio.ndim != 1:
        raise ValueError(f"audio must be 1-D or 2-D, got {audio.ndim}-D.")

    # Resample if needed
    if original_rate != target_rate:
        audio = resample_audio(audio, original_rate, target_rate)

    # Optional peak normalization
    if target_peak is not None:
        peak = np.max(np.abs(audio))
        if peak > np.finfo(np.float64).eps:
            audio = audio * (target_peak / peak)

    return audio, target_rate
