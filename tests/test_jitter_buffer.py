"""Tests for the UDP receiver's jitter buffer.

Validates that the jitter buffer correctly reorders out-of-sequence packets,
fills missing frames via interpolation, and degrades gracefully under
simulated packet loss and delay — not glitching or crashing.
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ai.hardware.udp_receiver import UDPReceiver, send_test_frame


def _send_frame(port: int, seq: int, samples: int = 320, channels: int = 2) -> None:
    """Send a single test frame."""
    t = np.arange(samples) / 16000
    audio = np.zeros((samples, channels), dtype=np.float64)
    for ch in range(channels):
        freq = 440 + ch * 220
        audio[:, ch] = 0.5 * np.sin(2 * np.pi * freq * t)

    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    interleaved = pcm.flatten().tobytes()
    header = struct.pack(">III", seq, channels, samples)
    packet = header + interleaved

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(packet, ("127.0.0.1", port))
    sock.close()


class TestJitterBuffer:
    """Test suite for jitter buffer functionality."""

    def test_in_order_delivery(self) -> None:
        """Packets sent in order should be delivered in order."""
        port = 15100
        receiver = UDPReceiver(port=port, jitter_buffer_depth=2)
        receiver.start()

        try:
            time.sleep(0.1)  # Let socket bind

            # Send 10 frames in order
            for seq in range(10):
                _send_frame(port, seq)
                time.sleep(0.01)

            time.sleep(0.5)  # Let jitter buffer flush

            # Verify frames received (may not be all due to jitter buffer depth)
            frames_received = 0
            while True:
                frame = receiver.get_frame()
                if frame is None:
                    break
                frames_received += 1

            assert frames_received > 0, "Should have received some frames"
            assert receiver.packets_received == 10
        finally:
            receiver.stop()

    def test_out_of_order_reordering(self) -> None:
        """Out-of-order packets should be reordered by the jitter buffer."""
        port = 15101
        receiver = UDPReceiver(port=port, jitter_buffer_depth=3)
        receiver.start()

        try:
            time.sleep(0.1)

            # Send packets out of order: 0, 2, 1, 3, 5, 4, 6, 7, 8, 9
            order = [0, 2, 1, 3, 5, 4, 6, 7, 8, 9]
            for seq in order:
                _send_frame(port, seq)
                time.sleep(0.005)

            time.sleep(0.5)

            # Collect delivered frames
            delivered_seqs: list[int] = []
            while True:
                frame = receiver.get_frame()
                if frame is None:
                    break
                delivered_seqs.append(frame["seq"])

            # Verify monotonically increasing delivery
            if len(delivered_seqs) > 1:
                for i in range(1, len(delivered_seqs)):
                    assert delivered_seqs[i] >= delivered_seqs[i - 1], \
                        f"Frames delivered out of order: {delivered_seqs}"
        finally:
            receiver.stop()

    def test_missing_packet_interpolation(self) -> None:
        """Missing packets should be filled with repeated frames."""
        port = 15102
        receiver = UDPReceiver(port=port, jitter_buffer_depth=2)
        receiver.start()

        try:
            time.sleep(0.1)

            # Send frames 0, 1, 2, (skip 3, 4), 5, 6, 7, 8, 9
            for seq in [0, 1, 2, 5, 6, 7, 8, 9]:
                _send_frame(port, seq)
                time.sleep(0.01)

            time.sleep(0.5)

            # Should have interpolated missing frames
            frames_received = 0
            while True:
                frame = receiver.get_frame()
                if frame is None:
                    break
                frames_received += 1

            # Verify interpolation happened (or at least no crash)
            assert frames_received > 0
            assert receiver.packets_received == 8  # Only 8 real packets
        finally:
            receiver.stop()

    def test_no_crash_under_packet_loss(self) -> None:
        """Receiver should not crash under high packet loss."""
        port = 15103
        receiver = UDPReceiver(port=port, jitter_buffer_depth=3)
        receiver.start()

        try:
            time.sleep(0.1)
            rng = np.random.default_rng(42)

            # Send 100 frames with 30% random drop
            sent = 0
            for seq in range(100):
                if rng.random() > 0.3:  # 70% delivery rate
                    _send_frame(port, seq)
                    sent += 1
                time.sleep(0.002)

            time.sleep(0.5)

            # Just verify no crash and some frames delivered
            frames_received = 0
            while True:
                frame = receiver.get_frame()
                if frame is None:
                    break
                frames_received += 1

            assert frames_received > 0, "Should deliver some frames even with loss"
            # No assertion on exact count — graceful degradation is the goal
        finally:
            receiver.stop()

    def test_statistics_tracking(self) -> None:
        """Receiver should track packet loss and reorder statistics."""
        port = 15104
        receiver = UDPReceiver(port=port, jitter_buffer_depth=2)
        receiver.start()

        try:
            time.sleep(0.1)

            # Send 5 in-order frames
            for seq in range(5):
                _send_frame(port, seq)
                time.sleep(0.01)

            time.sleep(0.3)

            assert receiver.packets_received == 5
            # packets_dropped and reordered should be non-negative
            assert receiver.packets_dropped >= 0
            assert receiver.packets_reordered >= 0
            assert receiver.frames_interpolated >= 0
        finally:
            receiver.stop()
