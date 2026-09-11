"""Abstract audio output interface and concrete implementations.

The processing engine depends **only** on :class:`AudioOutput` — it never
knows whether audio is going to a WAV file, laptop speakers, or Pi output.

Implementations
---------------
- :class:`WAVOutput` — writes enhanced audio to a WAV file
- :class:`LaptopSpeakerOutput` — plays through laptop speakers/headphones
- :class:`RaspberryPiOutput` — plays through Pi's audio output
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections import deque
from pathlib import Path

import numpy as np


class AudioOutput(ABC):
    """Abstract audio output sink."""

    @abstractmethod
    def open(self) -> None:
        """Prepare the output for writing."""

    @abstractmethod
    def write_frame(self, audio: np.ndarray) -> None:
        """Write one frame of enhanced audio.

        Parameters
        ----------
        audio : np.ndarray
            1-D float64 mono enhanced audio frame.
        """

    @abstractmethod
    def close(self) -> None:
        """Flush and release all resources."""

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""

    def __enter__(self) -> "AudioOutput":
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


# ======================================================================
# WAV file output
# ======================================================================

class WAVOutput(AudioOutput):
    """Writes enhanced audio to a WAV file.

    Accumulates all frames in memory and writes on ``close()``.

    Parameters
    ----------
    path : str or Path
        Output WAV file path.
    sample_rate : int
        Audio sample rate.
    """

    def __init__(
        self,
        path: str | Path,
        sample_rate: int = 16_000,
    ) -> None:
        self._path = Path(path)
        self._sr = sample_rate
        self._chunks: list[np.ndarray] = []

    def open(self) -> None:
        self._chunks = []
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write_frame(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float64).ravel()
        self._chunks.append(audio)

    def close(self) -> None:
        if not self._chunks:
            return
        from scipy.io import wavfile

        combined = np.concatenate(self._chunks)
        pcm = (combined * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(str(self._path), self._sr, pcm)
        self._chunks = []

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def total_samples(self) -> int:
        """Total samples written so far."""
        return sum(len(c) for c in self._chunks)


# ======================================================================
# Laptop speaker output
# ======================================================================

class LaptopSpeakerOutput(AudioOutput):
    """Plays enhanced audio through laptop speakers/headphones via sounddevice.

    Wraps the existing :class:`ai.hardware.playback.AudioPlayback` with
    the :class:`AudioOutput` interface.

    Parameters
    ----------
    sample_rate : int
        Output sample rate.
    block_size : int
        Output device block size.
    device_index : int, optional
        Specific output device.  ``None`` = system default.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        block_size: int = 512,
        device_index: int | None = None,
    ) -> None:
        self._sr = sample_rate
        self._block_size = block_size
        self._device_index = device_index
        self._playback = None

        # Stats
        self._frames_written = 0
        self._underruns = 0

    def open(self) -> None:
        from ai.hardware.playback import AudioPlayback

        self._playback = AudioPlayback(
            sample_rate=self._sr,
            channels=1,
            block_size=self._block_size,
        )
        self._playback.start()
        self._frames_written = 0

    def write_frame(self, audio: np.ndarray) -> None:
        if self._playback is not None:
            self._playback.enqueue(audio)
            self._frames_written += 1

    def close(self) -> None:
        if self._playback is not None:
            self._underruns = self._playback.underruns
            self._playback.stop()
            self._playback = None

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def underruns(self) -> int:
        if self._playback is not None:
            return self._playback.underruns
        return self._underruns

    def save_recording(self, path: str | Path) -> None:
        """Save all played audio to a WAV file (post-hoc recording)."""
        if self._playback is not None:
            self._playback.save_to_file(path)


# ======================================================================
# Raspberry Pi output
# ======================================================================

class RaspberryPiOutput(AudioOutput):
    """Plays audio through the Raspberry Pi's audio output.

    Uses PyAudio (ALSA) to write to the Pi's 3.5mm jack or USB DAC.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate.
    device_index : int, optional
        ALSA device index.  ``None`` = default output.
    buffer_frames : int
        Internal buffer size.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        device_index: int | None = None,
        buffer_frames: int = 200,
    ) -> None:
        self._sr = sample_rate
        self._device_index = device_index
        self._buffer_size = buffer_frames
        self._pa = None
        self._stream = None
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_frames)
        self._lock = threading.Lock()

    def open(self) -> None:
        try:
            import pyaudio
        except ImportError:
            raise ImportError(
                "pyaudio is required for Pi audio output. "
                "Install with: sudo apt-get install python3-pyaudio"
            )

        self._pa = pyaudio.PyAudio()

        stream_kwargs = {
            "format": pyaudio.paFloat32,
            "channels": 1,
            "rate": self._sr,
            "output": True,
            "frames_per_buffer": 512,
        }
        if self._device_index is not None:
            stream_kwargs["output_device_index"] = self._device_index

        self._stream = self._pa.open(**stream_kwargs)

    def write_frame(self, audio: np.ndarray) -> None:
        if self._stream is None:
            return
        audio = np.asarray(audio, dtype=np.float32).ravel()
        try:
            self._stream.write(audio.tobytes())
        except Exception:
            pass  # Best-effort on Pi

    def close(self) -> None:
        if self._stream is not None:
            self._stream.stop_stream()
            self._stream.close()
            self._stream = None
        if self._pa is not None:
            self._pa.terminate()
            self._pa = None

    @property
    def sample_rate(self) -> int:
        return self._sr


# ======================================================================
# Null output (for benchmarking / headless operation)
# ======================================================================

class NullOutput(AudioOutput):
    """Discards all audio — useful for benchmarking without hardware.

    Parameters
    ----------
    sample_rate : int
        Declared sample rate.
    """

    def __init__(self, sample_rate: int = 16_000) -> None:
        self._sr = sample_rate
        self._frames = 0
        self._samples = 0

    def open(self) -> None:
        self._frames = 0
        self._samples = 0

    def write_frame(self, audio: np.ndarray) -> None:
        self._frames += 1
        self._samples += len(np.asarray(audio).ravel())

    def close(self) -> None:
        pass

    @property
    def sample_rate(self) -> int:
        return self._sr

    @property
    def frames_written(self) -> int:
        return self._frames

    @property
    def total_samples(self) -> int:
        return self._samples
