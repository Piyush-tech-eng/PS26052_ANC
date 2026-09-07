"""Tests for UDP audio frame roundtrip (Phase 8)."""

from __future__ import annotations

import time

import numpy as np
import pytest

from ai.hardware.udp_receiver import UDPReceiver, send_test_frame


class TestUDPRoundtrip:
    """Loopback test: sender + receiver on localhost."""

    def test_single_frame_roundtrip(self) -> None:
        """Send one frame and verify it arrives intact."""
        port = 15005  # Use a non-standard port to avoid conflicts
        receiver = UDPReceiver(host="127.0.0.1", port=port)
        receiver.start()

        try:
            time.sleep(0.1)  # Let the listener start
            send_test_frame(host="127.0.0.1", port=port, seq=0,
                          samples=320, channels=2)
            time.sleep(0.2)  # Wait for receipt

            frame = receiver.get_frame()
            assert frame is not None, "No frame received"
            assert frame["seq"] == 0
            assert frame["channels"] == 2
            assert frame["audio"].shape == (320, 2)
            assert frame["audio"].dtype == np.float64
        finally:
            receiver.stop()

    def test_multiple_frames_ordering(self) -> None:
        """Multiple frames should arrive in order."""
        port = 15006
        receiver = UDPReceiver(host="127.0.0.1", port=port)
        receiver.start()

        try:
            time.sleep(0.1)
            num_frames = 5
            for i in range(num_frames):
                send_test_frame(host="127.0.0.1", port=port, seq=i,
                              samples=160, channels=1)
                time.sleep(0.01)

            time.sleep(0.3)

            received_seqs = []
            for _ in range(num_frames):
                frame = receiver.get_frame()
                if frame is not None:
                    received_seqs.append(frame["seq"])

            # At least most frames should arrive
            assert len(received_seqs) >= num_frames - 1
            # Should be in order
            assert received_seqs == sorted(received_seqs)
        finally:
            receiver.stop()

    def test_mono_downmix(self) -> None:
        """get_mono_frame should return 1-D float64."""
        port = 15007
        receiver = UDPReceiver(host="127.0.0.1", port=port)
        receiver.start()

        try:
            time.sleep(0.1)
            send_test_frame(host="127.0.0.1", port=port, seq=0,
                          samples=256, channels=2)
            time.sleep(0.2)

            mono = receiver.get_mono_frame()
            assert mono is not None
            assert mono.ndim == 1
            assert len(mono) == 256
            assert mono.dtype == np.float64
        finally:
            receiver.stop()

    def test_empty_buffer_returns_none(self) -> None:
        """get_frame on empty buffer returns None."""
        port = 15008
        receiver = UDPReceiver(host="127.0.0.1", port=port)
        receiver.start()

        try:
            time.sleep(0.1)
            assert receiver.get_frame() is None
            assert receiver.get_mono_frame() is None
        finally:
            receiver.stop()

    def test_packet_counting(self) -> None:
        """Receiver should track packet count."""
        port = 15009
        receiver = UDPReceiver(host="127.0.0.1", port=port)
        receiver.start()

        try:
            time.sleep(0.1)
            for i in range(3):
                send_test_frame(host="127.0.0.1", port=port, seq=i,
                              samples=160, channels=1)
                time.sleep(0.01)
            time.sleep(0.3)

            assert receiver.packets_received >= 3
        finally:
            receiver.stop()
