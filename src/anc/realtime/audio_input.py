"""Abstract audio input interface and concrete implementations.

The processing engine (``StreamingPipeline``, ``HybridEngine``) depends
**only** on :class:`AudioInput` — it never knows whether audio comes from
a WAV file, laptop microphone, or Raspberry Pi.

Implementations
---------------
- :class:`WAVInput` — reads stereo or mono WAV files frame-by-frame
- :class:`LaptopMicInput` — captures from the laptop's default microphone
- :class:`RaspberryPiInput` — receives two-channel audio from the Pi via UDP
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class AudioFrame:
    """One frame of audio from the input source.

    Attributes
    ----------
    reference : np.ndarray or None
        Reference microphone signal (noise-facing).  ``None`` when only
        a single mic is available (laptop loopback).
    error : np.ndarray
        Error / primary microphone signal (cancellation point).
    timestamp : float
        Wall-clock time when the frame was captured.
    sequence : int
        Monotonically increasing frame counter.
    """

    reference: np.ndarray | None
    error: np.ndarray
    timestamp: float = 0.0
    sequence: int = 0


class AudioInput(ABC):
    """Abstract audio source.

    Subclasses provide frames of audio in a uniform format regardless
    of the underlying hardware or file type.
    """

    @abstractmethod
    def open(self) -> None:
        """Prepare the source for reading (connect, open file, etc.)."""

    @abstractmethod
    def read_frame(self) -> AudioFrame | None:
        """Read the next audio frame.

        Returns ``None`` when the source is exhausted or no data is
        available yet (non-blocking).
        """

    @abstractmethod
    def close(self) -> None:
        """Release all resources."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Audio sample rate in Hz."""

    @property
    @abstractmethod
    def channels(self) -> int:
        """Number of audio channels (1 = mono, 2 = stereo ref+err)."""

    @property
    @abstractmethod
    def frame_size(self) -> int:
        """Number of samples per frame."""

    def __enter__(self) -> "AudioInput":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ======================================================================
# WAV file input
# ======================================================================

class WAVInput(AudioInput):
    """Reads a WAV file and yields frames at a configurable pace.

    Parameters
    ----------
    path : str or Path
        Path to the WAV file.
    frame_ms : int
        Frame duration in milliseconds (default 20).
    target_sample_rate : int
        If the file's sample rate differs, audio is resampled.
    realtime : bool
        If True, inserts sleep delays between frames to simulate
        real-time input.  If False, yields frames as fast as possible.
    reference_path : str or Path, optional
        Path to a separate WAV file for the reference channel.  If the
        main file is stereo, ``reference_channel`` and ``error_channel``
        are used instead.
    reference_channel : int
        Channel index for the reference signal (default 1).
    error_channel : int
        Channel index for the error signal (default 0).
    """

    def __init__(
        self,
        path: str | Path,
        frame_ms: int = 20,
        target_sample_rate: int = 16_000,
        realtime: bool = False,
        reference_path: str | Path | None = None,
        reference_channel: int = 1,
        error_channel: int = 0,
    ) -> None:
        self._path = Path(path)
        self._ref_path = Path(reference_path) if reference_path else None
        self._frame_ms = frame_ms
        self._target_sr = target_sample_rate
        self._realtime = realtime
        self._ref_ch = reference_channel
        self._err_ch = error_channel

        self._audio: np.ndarray | None = None
        self._ref_audio: np.ndarray | None = None
        self._pos = 0
        self._seq = 0

    def open(self) -> None:
        from scipy.io import wavfile

        sr, raw = wavfile.read(str(self._path))
        audio = self._normalize(raw)
        if sr != self._target_sr:
            audio = self._resample(audio, sr, self._target_sr)

        if audio.ndim == 2 and audio.shape[1] >= 2:
            self._audio = audio[:, self._err_ch]
            self._ref_audio = audio[:, self._ref_ch]
        else:
            if audio.ndim == 2:
                audio = audio[:, 0]
            self._audio = audio
            self._ref_audio = None

        # Separate reference file overrides channel splitting
        if self._ref_path is not None and self._ref_path.exists():
            sr_ref, raw_ref = wavfile.read(str(self._ref_path))
            ref = self._normalize(raw_ref)
            if ref.ndim > 1:
                ref = ref[:, 0]
            if sr_ref != self._target_sr:
                ref = self._resample(ref, sr_ref, self._target_sr)
            # Trim to match lengths
            min_len = min(len(self._audio), len(ref))
            self._audio = self._audio[:min_len]
            self._ref_audio = ref[:min_len]

        self._pos = 0
        self._seq = 0

    def read_frame(self) -> AudioFrame | None:
        if self._audio is None:
            return None

        fs = self.frame_size
        if self._pos >= len(self._audio):
            return None

        end = min(self._pos + fs, len(self._audio))
        err = self._audio[self._pos:end]
        if len(err) < fs:
            err = np.pad(err, (0, fs - len(err)))

        ref = None
        if self._ref_audio is not None:
            ref_end = min(self._pos + fs, len(self._ref_audio))
            ref = self._ref_audio[self._pos:ref_end]
            if len(ref) < fs:
                ref = np.pad(ref, (0, fs - len(ref)))

        frame = AudioFrame(
            reference=ref,
            error=err,
            timestamp=time.time(),
            sequence=self._seq,
        )

        self._pos += fs
        self._seq += 1

        if self._realtime:
            time.sleep(self._frame_ms / 1000.0)

        return frame

    def close(self) -> None:
        self._audio = None
        self._ref_audio = None
        self._pos = 0

    @property
    def sample_rate(self) -> int:
        return self._target_sr

    @property
    def channels(self) -> int:
        return 2 if self._ref_audio is not None else 1

    @property
    def frame_size(self) -> int:
        return int(self._target_sr * self._frame_ms / 1000)

    @staticmethod
    def _normalize(raw: np.ndarray) -> np.ndarray:
        if raw.dtype == np.int16:
            return raw.astype(np.float64) / 32768.0
        elif raw.dtype == np.float32:
            return raw.astype(np.float64)
        return raw.astype(np.float64)

    @staticmethod
    def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(sr_in, sr_out)
        if audio.ndim == 1:
            return resample_poly(audio, sr_out // g, sr_in // g)
        return np.column_stack([
            resample_poly(audio[:, ch], sr_out // g, sr_in // g)
            for ch in range(audio.shape[1])
        ])


# ======================================================================
# Laptop microphone input
# ======================================================================

class LaptopMicInput(AudioInput):
    """Captures from the laptop's default microphone via sounddevice.

    Single-channel: ``error = mic input``, ``reference = None``
    (classical ANC is disabled — AI-only enhancement).

    Parameters
    ----------
    sample_rate : int
        Desired sample rate.
    frame_ms : int
        Frame duration in milliseconds.
    device_index : int, optional
        Specific input device index.  ``None`` = system default.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        frame_ms: int = 20,
        device_index: int | None = None,
    ) -> None:
        self._sr = sample_rate
        self._frame_ms = frame_ms
        self._device_index = device_index
        self._queue: "queue.Queue[np.ndarray]" | None = None
        self._stream = None
        self._seq = 0

    def open(self) -> None:
        import queue
        import sounddevice as sd

        self._queue = queue.Queue(maxsize=200)
        self._seq = 0

        def _callback(indata, frames, time_info, status):
            if status:
                pass  # Could log
            self._queue.put(indata[:, 0].copy().astype(np.float64))

        self._stream = sd.InputStream(
            samplerate=self._sr,
            blocksize=self.frame_size,
            channels=1,
            dtype="float32",
            device=self._device_index,
            callback=_callback,
        )
        self._stream.start()

    def read_frame(self) -> AudioFrame | None:
        if self._queue is None:
            return None
        try:
            data = self._queue.get(timeout=0.1)
        except Exception:
            return None

        frame = AudioFrame(
            reference=None,
            error=data,
            timestamp=time.time(),
            sequence=self._seq,
        )
        self._seq += 1
        return frame

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._queue = None

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def channels(self) -> int:
        return 1

    @property
    def frame_size(self) -> int:
        return int(self._sr * self._frame_ms / 1000)


# ======================================================================
# Raspberry Pi input (UDP receiver wrapper)
# ======================================================================

class RaspberryPiInput(AudioInput):
    """Receives two-channel audio from the Pi via UDP.

    Wraps :class:`ai.hardware.udp_receiver.UDPReceiver` and labels the
    channels as reference and error according to the configuration.

    Parameters
    ----------
    host : str
        Bind address (default ``"0.0.0.0"``).
    port : int
        UDP port to listen on.
    sample_rate : int
        Expected audio sample rate.
    frame_ms : int
        Expected frame duration.
    reference_channel : int
        Which channel in the UDP packet is the reference mic (default 1).
    error_channel : int
        Which channel in the UDP packet is the error mic (default 0).
    jitter_buffer_depth : int
        Jitter buffer depth for reordering.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 5005,
        sample_rate: int = 16_000,
        frame_ms: int = 20,
        reference_channel: int = 1,
        error_channel: int = 0,
        jitter_buffer_depth: int = 1,
    ) -> None:
        self._host = host
        self._port = port
        self._sr = sample_rate
        self._frame_ms = frame_ms
        self._ref_ch = reference_channel
        self._err_ch = error_channel
        self._jitter_depth = jitter_buffer_depth
        self._receiver = None
        self._seq = 0

    def open(self) -> None:
        # Import here to avoid hard dependency in non-hardware profiles
        from ai.hardware.udp_receiver import UDPReceiver

        self._receiver = UDPReceiver(
            host=self._host,
            port=self._port,
            sample_rate=self._sr,
            jitter_buffer_depth=self._jitter_depth,
            reference_channel=self._ref_ch,
            error_channel=self._err_ch,
        )
        self._receiver.start()
        self._seq = 0

    def read_frame(self) -> AudioFrame | None:
        if self._receiver is None:
            return None

        stereo = self._receiver.get_stereo_frame()
        if stereo is None:
            return None

        frame = AudioFrame(
            reference=stereo["reference"],
            error=stereo["error"],
            timestamp=time.time(),
            sequence=stereo.get("seq", self._seq),
        )
        self._seq += 1
        return frame

    def close(self) -> None:
        if self._receiver is not None:
            self._receiver.stop()
            self._receiver = None

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def channels(self) -> int:
        return 2

    @property
    def frame_size(self) -> int:
        return int(self._sr * self._frame_ms / 1000)

    @property
    def receiver_stats(self) -> dict[str, int]:
        """Access underlying UDPReceiver statistics."""
        if self._receiver is None:
            return {}
        return {
            "packets_received": self._receiver.packets_received,
            "packets_dropped": self._receiver.packets_dropped,
            "packets_reordered": self._receiver.packets_reordered,
            "frames_interpolated": self._receiver.frames_interpolated,
            "buffer_level": self._receiver.buffer_level,
        }
