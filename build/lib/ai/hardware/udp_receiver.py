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
import time
from collections import deque
from typing import Any

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
    jitter_buffer_depth : int
        Number of frames to buffer for reordering (default ``1``).
    reference_channel : int
        Channel index for reference microphone (default ``1``).
    error_channel : int
        Channel index for error microphone (default ``0``).
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        buffer_size: int = 100,
        sample_rate: int = 16_000,
        jitter_buffer_depth: int = 1,
        reference_channel: int = 1,
        error_channel: int = 0,
    ) -> None:
        self._host = host
        self._port = port
        self._sample_rate = sample_rate
        self._jitter_depth = max(1, jitter_buffer_depth)
        self._reference_channel = reference_channel
        self._error_channel = error_channel
        self._channel_gains = np.ones(2, dtype=np.float64)

        # Thread-safe frame buffer
        self._buffer: deque[dict[str, np.ndarray | int]] = deque(
            maxlen=buffer_size
        )
        self._lock = threading.Lock()

        # Jitter buffer: holds packets for reordering before delivery
        self._jitter_buffer: dict[int, dict[str, np.ndarray | int]] = {}
        self._next_expected_seq: int | None = None
        self._last_delivered_frame: dict[str, np.ndarray | int] | None = None

        # Network state
        self._socket: socket.socket | None = None
        self._running = False
        self._thread: threading.Thread | None = None

        # Statistics
        self._packets_received = 0
        self._packets_dropped = 0
        self._packets_reordered = 0
        self._frames_interpolated = 0
        self._last_seq: int | None = None

    @property
    def packets_received(self) -> int:
        return self._packets_received

    @property
    def packets_dropped(self) -> int:
        return self._packets_dropped

    @property
    def packets_reordered(self) -> int:
        """Number of packets received out of order and reordered."""
        return self._packets_reordered

    @property
    def frames_interpolated(self) -> int:
        """Number of missing frames filled by interpolation/repetition."""
        return self._frames_interpolated

    @property
    def reference_channel(self) -> int:
        return self._reference_channel

    @reference_channel.setter
    def reference_channel(self, ch: int) -> None:
        self._reference_channel = ch

    @property
    def error_channel(self) -> int:
        return self._error_channel

    @error_channel.setter
    def error_channel(self, ch: int) -> None:
        self._error_channel = ch

    @property
    def channel_gains(self) -> np.ndarray:
        return self._channel_gains.copy()

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

    def get_frame(self) -> dict[str, Any] | None:
        """Pop the oldest frame from the buffer, or None if empty.

        Returns
        -------
        dict or None
            Keys: ``"audio"`` (np.ndarray, shape [samples, channels]),
            ``"seq"`` (int), ``"channels"`` (int), ``"flags"`` (int).
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

    def get_stereo_frame(self) -> dict[str, Any] | None:
        """Pop the oldest frame preserving both reference and error channels.

        Returns
        -------
        dict or None
            Keys:
              ``"reference"``: np.ndarray (float64, 1D reference mic signal)
              ``"error"``: np.ndarray (float64, 1D error/primary mic signal)
              ``"seq"``: int
              ``"raw_audio"``: np.ndarray (shape [samples, channels])
              ``"flags"``: int
        """
        frame = self.get_frame()
        if frame is None:
            return None

        audio = frame["audio"]
        flags = frame.get("flags", 0)

        # Apply channel gain normalization if 2-channel
        if audio.ndim == 2 and audio.shape[1] >= 2:
            if len(self._channel_gains) == audio.shape[1]:
                audio = audio * self._channel_gains

            # Check bit 0 of flags: if set, channel 0 is explicitly reference
            if flags & 1:
                ref_idx, err_idx = 0, 1
            else:
                ref_idx = min(self._reference_channel, audio.shape[1] - 1)
                err_idx = min(self._error_channel, audio.shape[1] - 1)

            ref = audio[:, ref_idx].copy()
            err = audio[:, err_idx].copy()
        elif audio.ndim == 2 and audio.shape[1] == 1:
            err = audio[:, 0].copy()
            ref = np.zeros_like(err)
        else:
            err = audio.copy()
            ref = np.zeros_like(err)

        return {
            "reference": ref,
            "error": err,
            "seq": frame.get("seq", 0),
            "raw_audio": audio,
            "flags": flags,
        }

    def calibrate_channels(self, duration_s: float = 2.0) -> dict[str, float]:
        """Calibrate channel gains based on ambient recording.

        Parameters
        ----------
        duration_s : float
            Duration in seconds to accumulate frames for calibration.

        Returns
        -------
        dict
            Calibration results including RMS values and gain adjustments.
        """
        frames: list[np.ndarray] = []
        t0 = time.time()
        while time.time() - t0 < duration_s:
            f = self.get_frame()
            if f is not None and f["audio"].ndim == 2 and f["audio"].shape[1] >= 2:
                frames.append(f["audio"])
            else:
                time.sleep(0.01)

        if not frames:
            return {"status": "no_data", "gain_ratio": 1.0}

        concatenated = np.vstack(frames)
        rms_0 = float(np.sqrt(np.mean(concatenated[:, 0] ** 2)))
        rms_1 = float(np.sqrt(np.mean(concatenated[:, 1] ** 2)))

        if rms_0 > 1e-6 and rms_1 > 1e-6:
            target_rms = (rms_0 + rms_1) / 2.0
            g0 = target_rms / rms_0
            g1 = target_rms / rms_1
            self._channel_gains = np.array([g0, g1], dtype=np.float64)
            gain_ratio = rms_1 / rms_0
        else:
            gain_ratio = 1.0

        return {
            "status": "calibrated",
            "rms_ch0": rms_0,
            "rms_ch1": rms_1,
            "gain_ratio": gain_ratio,
            "gain_ch0": float(self._channel_gains[0]),
            "gain_ch1": float(self._channel_gains[1]),
        }

    def _listen_loop(self) -> None:
        """Background thread: receive, parse, and jitter-buffer UDP packets."""
        while self._running:
            try:
                data, _ = self._socket.recvfrom(65535)  # type: ignore[union-attr]
            except socket.timeout:
                # Flush any mature packets from jitter buffer on timeout
                self._flush_jitter_buffer(force=True)
                continue
            except OSError:
                break

            if len(data) < 12:
                continue  # Malformed packet

            # Support both 16-byte (extended with flags) and 12-byte (legacy) headers
            flags = 0
            if len(data) >= 16:
                s_seq, s_ch, s_smp, s_flags = struct.unpack(">IIII", data[:16])
                if s_ch > 0 and s_smp > 0 and len(data) == 16 + (s_ch * s_smp * 2):
                    seq, channels, samples_per_ch, flags = s_seq, s_ch, s_smp, s_flags
                    payload = data[16:]
                else:
                    seq, channels, samples_per_ch = struct.unpack(">III", data[:12])
                    payload = data[12:]
            else:
                seq, channels, samples_per_ch = struct.unpack(">III", data[:12])
                payload = data[12:]

            # Check for dropped/reordered packets
            if self._last_seq is not None:
                expected = self._last_seq + 1
                if seq < expected:
                    self._packets_reordered += 1
                elif seq > expected:
                    self._packets_dropped += max(0, seq - expected)
            self._last_seq = seq

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
                "flags": flags,
            }

            # Add to jitter buffer for reordering
            self._jitter_buffer[seq] = frame
            self._packets_received += 1

            # Flush mature packets from jitter buffer
            self._flush_jitter_buffer()

    def _flush_jitter_buffer(self, force: bool = False) -> None:
        """Deliver packets in order from the jitter buffer.

        Waits until at least ``jitter_buffer_depth`` packets are buffered,
        then delivers the earliest ones in sequence order.  Missing packets
        trigger frame repetition (repeats the last successfully delivered
        frame) rather than silence gaps.
        """
        if not self._jitter_buffer:
            return

        if not force and len(self._jitter_buffer) < self._jitter_depth:
            return

        # Sort buffered sequence numbers
        buffered_seqs = sorted(self._jitter_buffer.keys())

        # Deliver packets up to (max_buffered - jitter_depth) or all if forced/depth=1
        if force or self._jitter_depth <= 1:
            deliver_up_to = buffered_seqs[-1]
        else:
            deliver_up_to = buffered_seqs[-self._jitter_depth]

        if self._next_expected_seq is None:
            self._next_expected_seq = buffered_seqs[0]

        for seq in range(self._next_expected_seq, deliver_up_to + 1):
            if seq in self._jitter_buffer:
                frame = self._jitter_buffer.pop(seq)
                with self._lock:
                    self._buffer.append(frame)
                self._last_delivered_frame = frame
            else:
                # Missing packet — repeat last frame (interpolation)
                if self._last_delivered_frame is not None:
                    interpolated = {
                        "audio": self._last_delivered_frame["audio"].copy(),
                        "seq": seq,
                        "channels": self._last_delivered_frame["channels"],
                    }
                    with self._lock:
                        self._buffer.append(interpolated)
                    self._frames_interpolated += 1

        self._next_expected_seq = deliver_up_to + 1

        # Clean up any stale entries below next_expected
        stale = [s for s in self._jitter_buffer if s < self._next_expected_seq]
        for s in stale:
            del self._jitter_buffer[s]

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
    flags: int = 0,
    use_extended_header: bool = False,
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
    flags : int
        Extended header flags (bit 0: ch0 is ref, bit 1: calibration).
    use_extended_header : bool
        If True or flags != 0, sends 16-byte header with flags.
    """
    t = np.arange(samples) / sample_rate
    audio = np.zeros((samples, channels), dtype=np.float64)
    for ch in range(channels):
        freq = 440 + ch * 220
        audio[:, ch] = 0.5 * np.sin(2 * np.pi * freq * t)

    # Encode as int16
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    interleaved = pcm.flatten().tobytes()

    if flags != 0 or use_extended_header:
        header = struct.pack(">IIII", seq, channels, samples, flags)
    else:
        header = struct.pack(">III", seq, channels, samples)
    packet = header + interleaved

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(packet, (host, port))
    sock.close()
