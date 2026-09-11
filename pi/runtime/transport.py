"""UDP transport layer for streaming Pi audio frames to the laptop."""

from __future__ import annotations

import socket
import struct
import numpy as np

from pi.runtime.config import PiConfig


class PiUDPSender:
    """Sends audio frames as sequenced UDP packets with extended header."""

    def __init__(self, config: PiConfig) -> None:
        self.config = config
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._seq = 0

    @property
    def sequence(self) -> int:
        return self._seq

    def send_frame(self, pcm_data: np.ndarray) -> None:
        """Package and send one audio frame."""
        header = struct.pack(
            ">IIII",
            self._seq,
            self.config.channels,
            self.config.samples_per_frame,
            self.config.flags,
        )
        packet = header + pcm_data.tobytes()
        try:
            self._sock.sendto(
                packet,
                (self.config.target_host, self.config.target_port),
            )
        except OSError:
            pass
        self._seq += 1

    def close(self) -> None:
        """Close socket."""
        try:
            self._sock.close()
        except Exception:
            pass
