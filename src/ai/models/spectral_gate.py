"""Spectral-gating speech enhancement model (zero external dependencies).

This is the **absolute fallback** denoiser — it requires only numpy/scipy
(already in the project's dependencies) and needs no pretrained weights,
model downloads, or GPU.  It implements noise-gate-style STFT-domain
suppression:

1.  Estimate the noise floor from the signal's spectral statistics.
2.  For each STFT frame, compute per-bin SNR relative to the noise floor.
3.  Apply a smooth gain mask that attenuates bins below the threshold while
    preserving bins above it.

The result is a mild but audible denoising that works on any signal and
validates the entire streaming + hardware pipeline end-to-end, independent
of RNNoise / DTLN availability.
"""

from __future__ import annotations

import numpy as np

from ai.models.base import EnhancementModel


class SpectralGateModel(EnhancementModel):
    """STFT spectral-gating denoiser with no external model dependency.

    Parameters
    ----------
    sample_rate : int
        Expected input sample rate in Hz (default 16000).
    fft_size : int
        FFT length (default 512).
    hop_size : int
        STFT hop length (default 128).
    noise_percentile : float
        Percentile of frame energies used to estimate the noise floor
        (default 25 — the quietest 25% of frames).
    threshold_db : float
        Spectral bins whose power is below ``noise_floor * threshold``
        are attenuated (default 6.0 dB above noise floor).
    min_gain_db : float
        Minimum gain applied to suppressed bins (default -20.0).
        Prevents total zeroing which sounds unnatural.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        fft_size: int = 512,
        hop_size: int = 128,
        noise_percentile: float = 25.0,
        threshold_db: float = 6.0,
        min_gain_db: float = -20.0,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        if fft_size <= 0 or hop_size <= 0:
            raise ValueError("fft_size and hop_size must be positive.")
        if not 0 < noise_percentile <= 100:
            raise ValueError("noise_percentile must be in (0, 100].")

        self._sample_rate = sample_rate
        self._fft_size = fft_size
        self._hop_size = hop_size
        self._noise_percentile = noise_percentile
        self._threshold_db = threshold_db
        self._min_gain_linear = 10.0 ** (min_gain_db / 20.0)

        # Running noise-floor estimate for streaming use
        self._noise_floor: np.ndarray | None = None
        self._noise_frame_count: int = 0

    # ------------------------------------------------------------------
    # EnhancementModel interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "spectral_gate"

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def frame_size(self) -> int:
        # Can handle arbitrary lengths via internal STFT framing
        return 0

    def enhance(self, x: np.ndarray) -> np.ndarray:
        """Apply spectral gating to denoise *x*.

        Parameters
        ----------
        x : np.ndarray
            Mono float64 audio at ``self.sample_rate``.

        Returns
        -------
        np.ndarray
            Denoised audio, same length as *x*.
        """
        x = self.validate_input(x)
        original_length = len(x)

        # ----- STFT analysis -----
        # Use scipy-style centered STFT with proper OLA reconstruction
        fft = self._fft_size
        hop = self._hop_size

        # Hann window
        window = np.hanning(fft).astype(np.float64)

        # Pad signal so we get complete coverage
        # Pad at least half a window on each side for centering
        pad_left = fft // 2
        pad_right = fft  # generous right pad
        x_padded = np.pad(x, (pad_left, pad_right), mode="reflect")

        num_frames = 1 + (len(x_padded) - fft) // hop
        num_bins = fft // 2 + 1
        floor = np.finfo(np.float64).eps

        # Forward STFT
        magnitudes = np.zeros((num_frames, num_bins), dtype=np.float64)
        phases = np.zeros((num_frames, num_bins), dtype=np.float64)

        for i in range(num_frames):
            start = i * hop
            frame = x_padded[start:start + fft] * window
            spectrum = np.fft.rfft(frame)
            magnitudes[i] = np.abs(spectrum)
            phases[i] = np.angle(spectrum)

        # ----- Noise floor estimation -----
        # Compute per-frame total power, use quiet frames to estimate noise
        frame_powers = np.sum(magnitudes ** 2, axis=1)

        # Use the specified percentile of frames as noise reference
        percentile_threshold = np.percentile(
            frame_powers, self._noise_percentile
        )
        noise_mask = frame_powers <= percentile_threshold
        noise_frame_count = max(1, np.sum(noise_mask))

        if self._noise_floor is None:
            # Per-bin mean power from the quietest frames
            self._noise_floor = np.mean(
                magnitudes[noise_mask] ** 2, axis=0
            )
            self._noise_frame_count = int(noise_frame_count)
        else:
            new_estimate = np.mean(magnitudes[noise_mask] ** 2, axis=0)
            alpha = 0.8
            self._noise_floor = (
                alpha * self._noise_floor + (1 - alpha) * new_estimate
            )
            self._noise_frame_count += int(noise_frame_count)

        noise_floor = np.maximum(self._noise_floor, floor)

        # ----- Per-frame spectral gating -----
        threshold_mult = 10.0 ** (self._threshold_db / 10.0)
        gate_threshold = noise_floor * threshold_mult

        gains = np.ones_like(magnitudes)
        for i in range(num_frames):
            frame_power = magnitudes[i] ** 2

            # Soft gain: Wiener-like filter
            # gain = max(min_gain, 1 - noise_floor / frame_power)
            # This smoothly transitions from ~1 (clean) to min_gain (noisy)
            snr = frame_power / (noise_floor + floor)
            # Wiener gain: (SNR - 1) / SNR, clamped
            wiener_gain = np.maximum(0.0, 1.0 - 1.0 / (snr + floor))
            # Apply threshold: below threshold -> min_gain
            below_threshold = frame_power < gate_threshold
            wiener_gain[below_threshold] = self._min_gain_linear
            # Ensure minimum gain
            wiener_gain = np.maximum(wiener_gain, self._min_gain_linear)
            gains[i] = wiener_gain

        # ----- Inverse STFT (overlap-add) -----
        output_length = (num_frames - 1) * hop + fft
        output = np.zeros(output_length, dtype=np.float64)
        window_sum = np.zeros(output_length, dtype=np.float64)

        for i in range(num_frames):
            start = i * hop
            # Apply gain in spectral domain
            enhanced_spectrum = (
                gains[i] * magnitudes[i] * np.exp(1j * phases[i])
            )
            frame = np.fft.irfft(enhanced_spectrum, n=fft).real
            frame *= window
            output[start:start + fft] += frame
            window_sum[start:start + fft] += window ** 2

        # Normalize by window overlap
        nonzero = window_sum > floor
        output[nonzero] /= window_sum[nonzero]

        # Remove padding and return original length
        result = output[pad_left:pad_left + original_length]
        return result.copy()

    def reset(self) -> None:
        """Reset the running noise floor estimate for a new stream."""
        self._noise_floor = None
        self._noise_frame_count = 0
