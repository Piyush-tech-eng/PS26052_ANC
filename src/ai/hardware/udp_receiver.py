"""UDP audio frame receiver for the laptop side of the Pi→Laptop pipeline.

Listens for UDP packets from ``pi/capture_stream.py``, reassembles incoming
PCM frames per channel into a thread-safe ring buffer, and feeds them to the
``HybridEngine`` for real-time processing.

Protocol
--------
Each UDP packet contains:
- 4 bytes: sequence number (uint32, big-endian)
- 4 bytes: channel count (uint32, big-endian)
- 4 bytes: samples per channel (uint32, big-endian)
- remaining: interleaved int16 PCM samples (little-endian)
"""

from __future__ import annotations

import socket
import struct
import threading
from collections import deque

import numpy as np


class UDPReceiver:
    """Laptop-side UDP listener that reassembles audio frames.

    Parameters
    ----------
    host : str
        Bind address (default ``"0.0.0.0"`` — all interfaces).
    port : int
        UDP port to listen on (default ``5005``).
    buffer_size : int
        Maximum number of frames to buffer (default ``100``).
    sample_rate : int
        Expected sample rate (for metadata, default ``16000``).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        buffer_size: int = 100,
        sample_rate: int = 16_000,
    ) -> None:
        self._host = host
        self._port = port
        self._sample_rate = sample_rate

        # Thread-safe frame buffer
        self._buffer: deque[dict[str, np.ndarray | int]] = deque(
            maxlen=buffer_size
        )
        self._lock = threading.Lock()

        # Network state
        self._socket: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        # Statistics
        self._packets_received = 0
        self._packets_dropped = 0
        self._last_seq: int | None = None

    @property
    def packets_received(self) -> int:
        return self._packets_received

    @property
    def packets_dropped(self) -> int:
        return self._packets_dropped

    @property
    def buffer_level(self) -> int:
        """Number of frames currently buffered."""
        with self._lock:
            return len(self._buffer)

    def start(self) -> None:
        """Start the UDP listener in a background thread."""
        if self._running:
            return

        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self._host, self._port))
        self._socket.settimeout(0.5)  # For clean shutdown

        self._running = True
        self._thread = threading.Thread(
            target=self._listen_loop, daemon=True, name="udp-receiver"
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the listener and close the socket."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def get_frame(self) -> dict[str, np.ndarray | int] | None:
        """Pop the oldest frame from the buffer, or None if empty.

        Returns
        -------
        dict or None
            Keys: ``"audio"`` (np.ndarray, shape [samples, channels]),
            ``"seq"`` (int), ``"channels"`` (int).
        """
        with self._lock:
            if self._buffer:
                return self._buffer.popleft()
        return None

    def get_mono_frame(self) -> np.ndarray | None:
        """Pop the oldest frame and downmix to mono float64.

        Returns None if no frames are available.
        """
        frame = self.get_frame()
        if frame is None:
            return None
        audio = frame["audio"]
        if audio.ndim == 2:
            return np.mean(audio, axis=1).astype(np.float64)
        return audio.astype(np.float64)

    def _listen_loop(self) -> None:
        """Background thread: receive and parse UDP packets."""
        while self._running:
            try:
                data, _ = self._socket.recvfrom(65535)  # type: ignore[union-attr]
            except socket.timeout:
                continue
            except OSError:
                break

            if len(data) < 12:
                continue  # Malformed packet

            # Parse header
            seq, channels, samples_per_ch = struct.unpack(">III", data[:12])

            # Check for dropped packets
            if self._last_seq is not None:
                expected = self._last_seq + 1
                if seq != expected:
                    self._packets_dropped += max(0, seq - expected)
            self._last_seq = seq

            # Parse PCM payload
            payload = data[12:]
            total_samples = channels * samples_per_ch
            expected_bytes = total_samples * 2  # int16

            if len(payload) < expected_bytes:
                continue  # Truncated packet

            pcm_int16 = np.frombuffer(
                payload[:expected_bytes], dtype=np.dtype("<i2")
            )
            # De-interleave and normalize to float64
            audio = pcm_int16.reshape(-1, channels).astype(np.float64) / 32768.0

            frame = {
                "audio": audio,
                "seq": seq,
                "channels": channels,
            }

            with self._lock:
                self._buffer.append(frame)
            self._packets_received += 1

    def __enter__(self) -> "UDPReceiver":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()


def send_test_frame(
    host: str = "127.0.0.1",
    port: int = 5005,
    seq: int = 0,
    samples: int = 320,
    channels: int = 2,
    sample_rate: int = 16_000,
) -> None:
    """Send a single test frame via UDP (for loopback testing).

    Parameters
    ----------
    host : str
        Target address.
    port : int
        Target UDP port.
    seq : int
        Sequence number.
    samples : int
        Samples per channel.
    channels : int
        Number of audio channels.
    sample_rate : int
        Sample rate (for generating test tone).
    """
    t = np.arange(samples) / sample_rate
    audio = np.zeros((samples, channels), dtype=np.float64)
    for ch in range(channels):
        freq = 440 + ch * 220
        audio[:, ch] = 0.5 * np.sin(2 * np.pi * freq * t)

    # Encode as int16
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    interleaved = pcm.flatten().tobytes()

    header = struct.pack(">III", seq, channels, samples)
    packet = header + interleaved

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(packet, (host, port))
    sock.close()
