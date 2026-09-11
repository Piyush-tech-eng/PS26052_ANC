"""Unit tests for two-microphone channel calibration and validation."""

from __future__ import annotations

import numpy as np
import pytest

from anc.calibration.channel_test import ChannelCalibrator, CalibrationResult


class TestChannelCalibrator:
    """Unit tests for ChannelCalibrator."""

    def test_ideal_signals_calibration(self) -> None:
        sr = 16_000
        t = np.arange(sr) / sr
        # Correlated reference and error (reference louder)
        noise = 0.5 * np.sin(2 * np.pi * 300 * t)
        ref = noise
        err = 0.5 * noise  # Error is attenuated version (gain ratio = 2.0)

        calibrator = ChannelCalibrator()
        res = calibrator.calibrate_from_arrays(ref, err, sample_rate=sr)

        assert res.valid
        assert res.channels_correlated
        assert res.channels_synchronized
        assert res.channels_ordered_correctly
        assert np.isclose(res.gain_ratio, 2.0, atol=0.05)
        assert res.correlation_peak > 0.9
        assert "[OK] CALIBRATION PASSED" in res.summary

    def test_delayed_signals_detection(self) -> None:
        sr = 16_000
        t = np.arange(sr) / sr
        noise = 0.5 * np.sin(2 * np.pi * 300 * t)

        delay_samples = 8
        ref = noise
        err = np.roll(noise * 0.4, delay_samples)

        calibrator = ChannelCalibrator()
        res = calibrator.calibrate_from_arrays(ref, err, sample_rate=sr)

        assert res.channels_correlated
        assert abs(res.delay_samples) <= 10
        assert abs(res.delay_ms) <= 5.0

    def test_channel_swap_detection(self) -> None:
        sr = 16_000
        t = np.arange(sr) / sr
        noise = 0.5 * np.sin(2 * np.pi * 300 * t)
        # Inverted: reference is weaker than error
        ref = 0.1 * noise
        err = 0.5 * noise

        calibrator = ChannelCalibrator()
        res = calibrator.calibrate_from_arrays(ref, err, sample_rate=sr)

        assert not res.channels_ordered_correctly
        assert res.gain_ratio < 0.5
        assert "Reference mic has lower RMS than error mic" in res.summary

    def test_silent_channel_handling(self) -> None:
        sr = 16_000
        ref = np.zeros(sr)
        err = np.zeros(sr)

        calibrator = ChannelCalibrator()
        res = calibrator.calibrate_from_arrays(ref, err, sample_rate=sr)

        assert not res.valid
        assert not res.channels_correlated
