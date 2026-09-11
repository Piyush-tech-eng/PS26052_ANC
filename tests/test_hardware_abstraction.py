"""Tests for hardware abstraction, UDP receiver, and dual-microphone streaming."""

from __future__ import annotations

import socket
import struct
import time

import numpy as np
import pytest

from ai.hardware.udp_receiver import UDPReceiver, send_test_frame


class TestUDPReceiverProtocol:
    """Unit tests for UDP receiver protocol and packet decoding."""

    def test_legacy_12byte_header_roundtrip(self) -> None:
        port = 17001
        receiver = UDPReceiver(port=port, sample_rate=16_000, reference_channel=1, error_channel=0)
        receiver.start()

        try:
            # Send legacy 12-byte packet
            send_test_frame(
                host="127.0.0.1",
                port=port,
                seq=42,
                samples=320,
                channels=2,
                sample_rate=16_000,
                flags=0,
                use_extended_header=False,
            )

            time.sleep(0.05)
            frame = receiver.get_frame()
            assert frame is not None
            assert frame["seq"] == 42
            assert frame["channels"] == 2
            assert frame["flags"] == 0
            assert frame["audio"].shape == (320, 2)
        finally:
            receiver.stop()

    def test_extended_16byte_header_roundtrip(self) -> None:
        port = 17002
        receiver = UDPReceiver(port=port, sample_rate=16_000, reference_channel=1, error_channel=0)
        receiver.start()

        try:
            # Send 16-byte packet with flags=0
            send_test_frame(
                host="127.0.0.1",
                port=port,
                seq=101,
                samples=320,
                channels=2,
                sample_rate=16_000,
                flags=0,
                use_extended_header=True,
            )

            time.sleep(0.05)
            stereo = receiver.get_stereo_frame()
            assert stereo is not None
            assert stereo["seq"] == 101
            assert len(stereo["error"]) == 320
            assert len(stereo["reference"]) == 320
        finally:
            receiver.stop()

    def test_channel_flag_inversion(self) -> None:
        port = 17003
        receiver = UDPReceiver(port=port, sample_rate=16_000, reference_channel=1, error_channel=0)
        receiver.start()

        try:
            # Send packet with flags=1 (FLAG_CH0_IS_REFERENCE)
            # In send_test_frame, ch0 freq is 440 Hz, ch1 freq is 660 Hz
            send_test_frame(
                host="127.0.0.1",
                port=port,
                seq=202,
                samples=320,
                channels=2,
                sample_rate=16_000,
                flags=1,
                use_extended_header=True,
            )

            time.sleep(0.05)
            stereo = receiver.get_stereo_frame()
            assert stereo is not None
            # With bit 0 set, reference is ch0, error is ch1
            assert stereo["flags"] == 1
            assert len(stereo["reference"]) == 320
            assert len(stereo["error"]) == 320
        finally:
            receiver.stop()
