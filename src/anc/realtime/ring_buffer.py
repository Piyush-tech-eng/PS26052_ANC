"""Thread-safe ring buffer for real-time audio streaming.

Provides a fixed-capacity circular buffer with separate write (producer)
and read (consumer) cursors, overflow/underflow detection, and health
metrics for the streaming pipeline telemetry.

Usage::

    buf = AudioRingBuffer(capacity_samples=32000, channels=2)
    buf.write(frame_reference, frame_error)
    ref, err = buf.read(320)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


@dataclass
class BufferHealth:
    """Snapshot of ring buffer health metrics."""

    capacity_samples: int
    fill_level: int
    fill_ratio: float
    overflows: int
    underflows: int
    total_written: int
    total_read: int


class AudioRingBuffer:
    """Thread-safe circular buffer for two-channel (reference + error) audio.

    Parameters
    ----------
    capacity_samples : int
        Maximum number of samples the buffer can hold per channel.
    channels : int
        Number of audio channels (default 2: reference + error).
    overflow_policy : str
        ``"drop_oldest"`` — overwrite oldest data on overflow.
        ``"drop_newest"`` — discard incoming data on overflow.
    """

    def __init__(
        self,
        capacity_samples: int = 32_000,
        channels: int = 2,
        overflow_policy: str = "drop_oldest",
    ) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive.")
        if channels <= 0:
            raise ValueError("channels must be positive.")
        if overflow_policy not in ("drop_oldest", "drop_newest"):
            raise ValueError(f"Unknown overflow_policy: {overflow_policy}")

        self._capacity = capacity_samples
        self._channels = channels
        self._overflow_policy = overflow_policy

        # Storage: [channels, capacity]
        self._data = np.zeros((channels, capacity_samples), dtype=np.float64)

        # Cursors
        self._write_pos = 0
        self._read_pos = 0
        self._count = 0  # Number of valid samples

        self._lock = threading.Lock()

        # Stats
        self._overflows = 0
        self._underflows = 0
        self._total_written = 0
        self._total_read = 0

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def channels(self) -> int:
        return self._channels

    def fill_level(self) -> int:
        """Number of samples available to read."""
        with self._lock:
            return self._count

    def fill_ratio(self) -> float:
        """Fill ratio as a fraction of capacity (0.0 – 1.0)."""
        with self._lock:
            return self._count / self._capacity

    def health(self) -> BufferHealth:
        """Current buffer health snapshot."""
        with self._lock:
            return BufferHealth(
                capacity_samples=self._capacity,
                fill_level=self._count,
                fill_ratio=self._count / self._capacity,
                overflows=self._overflows,
                underflows=self._underflows,
                total_written=self._total_written,
                total_read=self._total_read,
            )

    def write(self, *channel_data: np.ndarray) -> int:
        """Write samples to the ring buffer.

        Parameters
        ----------
        *channel_data : np.ndarray
            One array per channel, all same length.  For two-channel mode,
            call as ``write(reference, error)``.

        Returns
        -------
        int
            Number of samples actually written.
        """
        if len(channel_data) != self._channels:
            raise ValueError(
                f"Expected {self._channels} channel arrays, got {len(channel_data)}."
            )

        arrays = [np.asarray(ch, dtype=np.float64).ravel() for ch in channel_data]
        n_samples = len(arrays[0])
        if any(len(a) != n_samples for a in arrays):
            raise ValueError("All channel arrays must have the same length.")

        if n_samples == 0:
            return 0

        with self._lock:
            space = self._capacity - self._count

            if n_samples > space:
                if self._overflow_policy == "drop_newest":
                    # Only write what fits
                    n_samples = space
                    if n_samples == 0:
                        self._overflows += 1
                        return 0
                    arrays = [a[:n_samples] for a in arrays]
                else:
                    # drop_oldest: advance read pointer to make room
                    overflow_amount = n_samples - space
                    self._read_pos = (self._read_pos + overflow_amount) % self._capacity
                    self._count -= overflow_amount
                    self._overflows += 1

            # Write into the circular buffer
            end_pos = self._write_pos + n_samples
            if end_pos <= self._capacity:
                for ch_idx, arr in enumerate(arrays):
                    self._data[ch_idx, self._write_pos:end_pos] = arr
            else:
                # Wrap around
                first_chunk = self._capacity - self._write_pos
                for ch_idx, arr in enumerate(arrays):
                    self._data[ch_idx, self._write_pos:] = arr[:first_chunk]
                    self._data[ch_idx, :n_samples - first_chunk] = arr[first_chunk:]

            self._write_pos = (self._write_pos + n_samples) % self._capacity
            self._count += n_samples
            self._total_written += n_samples

            return n_samples

    def read(self, n_samples: int) -> tuple[np.ndarray, ...] | None:
        """Read samples from the ring buffer.

        Parameters
        ----------
        n_samples : int
            Number of samples to read per channel.

        Returns
        -------
        tuple of np.ndarray or None
            One array per channel, or None if fewer than ``n_samples``
            are available (underflow).
        """
        with self._lock:
            if self._count < n_samples:
                self._underflows += 1
                return None

            result = []
            end_pos = self._read_pos + n_samples
            if end_pos <= self._capacity:
                for ch_idx in range(self._channels):
                    result.append(
                        self._data[ch_idx, self._read_pos:end_pos].copy()
                    )
            else:
                first_chunk = self._capacity - self._read_pos
                for ch_idx in range(self._channels):
                    part1 = self._data[ch_idx, self._read_pos:]
                    part2 = self._data[ch_idx, :n_samples - first_chunk]
                    result.append(np.concatenate([part1, part2]))

            self._read_pos = (self._read_pos + n_samples) % self._capacity
            self._count -= n_samples
            self._total_read += n_samples

            return tuple(result)

    def read_available(self, max_samples: int = 0) -> tuple[np.ndarray, ...] | None:
        """Read all available samples, up to *max_samples* per channel.

        Returns None if the buffer is empty.
        """
        with self._lock:
            if self._count == 0:
                return None
            n = min(self._count, max_samples) if max_samples > 0 else self._count

        return self.read(n)

    def clear(self) -> None:
        """Discard all buffered data."""
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._count = 0
