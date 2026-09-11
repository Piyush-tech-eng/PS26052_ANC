"""Startup channel calibration and test mode.

Before starting the main processing loop, a calibration run:

1. Captures N seconds of ambient audio from both microphones.
2. Computes per-channel RMS and gain ratio.
3. Measures cross-correlation delay between channels.
4. Validates reference-to-error correlation (confirms channels are
   capturing the same acoustic environment, not swapped).
5. Reports channel ordering and synchronization status.

Usage::

    from anc.calibration.channel_test import ChannelCalibrator
    from anc.realtime.audio_input import RaspberryPiInput

    source = RaspberryPiInput(port=5005)
    cal = ChannelCalibrator()
    result = cal.run_calibration(source, duration_s=3.0)
    print(result.summary)
    if not result.valid:
        print("WARNING: Channel calibration failed — check mic connections")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class CalibrationResult:
    """Result of a channel calibration run."""

    # Per-channel measurements
    reference_rms: float = 0.0
    error_rms: float = 0.0
    gain_ratio: float = 1.0          # reference_rms / error_rms

    # Cross-correlation
    delay_samples: int = 0           # Delay of error relative to reference
    delay_ms: float = 0.0
    correlation_peak: float = 0.0    # Peak cross-correlation coefficient

    # Validation
    channels_synchronized: bool = False
    channels_correlated: bool = False
    channels_ordered_correctly: bool = False
    valid: bool = False

    # Raw data for debugging
    sample_rate: int = 16_000
    duration_s: float = 0.0
    frames_captured: int = 0

    @property
    def summary(self) -> str:
        """Human-readable calibration summary."""
        lines = [
            "Channel Calibration Results",
            "=" * 40,
            f"  Duration:      {self.duration_s:.1f}s ({self.frames_captured} frames)",
            f"  Sample rate:   {self.sample_rate} Hz",
            "",
            "  Reference mic (noise-facing):",
            f"    RMS:         {self.reference_rms:.4f}",
            "  Error mic (cancellation point):",
            f"    RMS:         {self.error_rms:.4f}",
            f"  Gain ratio:    {self.gain_ratio:.3f} (ref/err)",
            "",
            f"  Cross-correlation delay: {self.delay_samples} samples ({self.delay_ms:.2f} ms)",
            f"  Correlation peak:        {self.correlation_peak:.4f}",
            "",
            f"  Synchronized:   {'[OK]' if self.channels_synchronized else '[X]'}",
            f"  Correlated:     {'[OK]' if self.channels_correlated else '[X]'}",
            f"  Correctly ordered: {'[OK]' if self.channels_ordered_correctly else '[X]'}",
            "",
        ]
        if self.valid:
            lines.append("  [OK] CALIBRATION PASSED")
        else:
            lines.append("  [X] CALIBRATION ISSUES DETECTED")
            if not self.channels_correlated:
                lines.append("    -> Channels show low correlation — mics may not be capturing the same environment")
            if not self.channels_synchronized:
                lines.append(f"    -> Channel delay of {self.delay_ms:.1f} ms exceeds synchronization threshold")
            if not self.channels_ordered_correctly:
                lines.append("    -> Reference mic has lower RMS than error mic — channels may be swapped")
        return "\n".join(lines)


class ChannelCalibrator:
    """Two-microphone calibration and validation.

    Parameters
    ----------
    correlation_threshold : float
        Minimum cross-correlation peak to consider channels as
        capturing the same acoustic environment (default 0.3).
    max_delay_ms : float
        Maximum acceptable delay between channels in milliseconds
        (default 5.0).
    min_rms : float
        Minimum RMS to consider a channel as actively capturing
        audio (default 0.001).
    """

    def __init__(
        self,
        correlation_threshold: float = 0.3,
        max_delay_ms: float = 5.0,
        min_rms: float = 0.001,
    ) -> None:
        self._corr_threshold = correlation_threshold
        self._max_delay_ms = max_delay_ms
        self._min_rms = min_rms

    def run_calibration(
        self,
        input_source,
        duration_s: float = 3.0,
    ) -> CalibrationResult:
        """Run the calibration procedure.

        Parameters
        ----------
        input_source : AudioInput
            An opened audio input source (must be two-channel).
        duration_s : float
            Duration to capture for calibration.

        Returns
        -------
        CalibrationResult
            Detailed calibration outcome.
        """
        import time

        # Collect frames
        ref_chunks: list[np.ndarray] = []
        err_chunks: list[np.ndarray] = []
        frames_captured = 0
        sample_rate = input_source.sample_rate

        start = time.time()
        while time.time() - start < duration_s:
            frame = input_source.read_frame()
            if frame is None:
                time.sleep(0.005)
                continue

            err_chunks.append(frame.error)
            if frame.reference is not None:
                ref_chunks.append(frame.reference)
            frames_captured += 1

        if not err_chunks or not ref_chunks:
            return CalibrationResult(
                sample_rate=sample_rate,
                duration_s=time.time() - start,
                frames_captured=frames_captured,
                valid=False,
            )

        # Concatenate all captured audio
        reference = np.concatenate(ref_chunks)
        error = np.concatenate(err_chunks)

        # Trim to equal lengths
        min_len = min(len(reference), len(error))
        reference = reference[:min_len]
        error = error[:min_len]

        return self.calibrate_from_arrays(
            reference, error,
            sample_rate=sample_rate,
            frames_captured=frames_captured,
        )

    def calibrate_from_arrays(
        self,
        reference: np.ndarray,
        error: np.ndarray,
        sample_rate: int = 16_000,
        frames_captured: int = 0,
    ) -> CalibrationResult:
        """Run calibration analysis on pre-captured arrays.

        Useful for offline testing without hardware.
        """
        reference = np.asarray(reference, dtype=np.float64).ravel()
        error = np.asarray(error, dtype=np.float64).ravel()

        duration_s = len(reference) / sample_rate

        # Per-channel RMS
        ref_rms = float(np.sqrt(np.mean(reference ** 2)))
        err_rms = float(np.sqrt(np.mean(error ** 2)))
        gain_ratio = ref_rms / (err_rms + 1e-10)

        # Cross-correlation to measure delay
        max_lag = int(self._max_delay_ms * sample_rate / 1000) * 2
        max_lag = min(max_lag, len(reference) // 2)

        # Normalized cross-correlation
        ref_norm = reference - np.mean(reference)
        err_norm = error - np.mean(error)
        ref_std = np.std(ref_norm)
        err_std = np.std(err_norm)

        if ref_std < 1e-10 or err_std < 1e-10:
            # One or both channels are silent
            return CalibrationResult(
                reference_rms=ref_rms,
                error_rms=err_rms,
                gain_ratio=gain_ratio,
                sample_rate=sample_rate,
                duration_s=duration_s,
                frames_captured=frames_captured,
                valid=False,
            )

        # Full cross-correlation
        correlation = np.correlate(
            ref_norm[:max_lag * 2],
            err_norm[:max_lag * 2],
            mode="full",
        )
        # Normalize
        correlation /= (len(ref_norm[:max_lag * 2]) * ref_std * err_std + 1e-10)

        peak_idx = int(np.argmax(np.abs(correlation)))
        center = len(err_norm[:max_lag * 2]) - 1
        delay_samples = peak_idx - center
        delay_ms = delay_samples / sample_rate * 1000.0
        correlation_peak = float(np.abs(correlation[peak_idx]))

        # Validation checks
        channels_synchronized = abs(delay_ms) <= self._max_delay_ms
        channels_correlated = correlation_peak >= self._corr_threshold
        # In a feedforward setup, the reference mic (noise-facing) should
        # generally have stronger noise signal -> higher or similar RMS
        channels_ordered_correctly = ref_rms >= err_rms * 0.5  # Allow 6dB tolerance
        valid = channels_synchronized and channels_correlated and channels_ordered_correctly

        return CalibrationResult(
            reference_rms=ref_rms,
            error_rms=err_rms,
            gain_ratio=gain_ratio,
            delay_samples=delay_samples,
            delay_ms=delay_ms,
            correlation_peak=correlation_peak,
            channels_synchronized=channels_synchronized,
            channels_correlated=channels_correlated,
            channels_ordered_correctly=channels_ordered_correctly,
            valid=valid,
            sample_rate=sample_rate,
            duration_s=duration_s,
            frames_captured=frames_captured,
        )
