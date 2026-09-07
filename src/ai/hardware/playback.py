"""Real-time audio playback via sounddevice callback-based streaming.

Pulls enhanced frames from the ``HybridEngine`` and outputs them through
the system's default audio device.  Falls back to a file-writing mode if
``sounddevice`` is not available.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np


def _sounddevice_available() -> bool:
    try:
        import sounddevice  # noqa: F401
        return True
    except (ImportError, OSError):
        return False


class AudioPlayback:
    """Real-time audio output using sounddevice or file fallback.

    Parameters
    ----------
    sample_rate : int
        Output sample rate in Hz.
    channels : int
        Number of output channels (default 1 = mono).
    buffer_size : int
        Maximum number of frames to buffer before dropping.
    block_size : int
        Number of samples per output block (default 512).
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        channels: int = 1,
        buffer_size: int = 200,
        block_size: int = 512,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        self._sample_rate = sample_rate
        self._channels = channels
        self._block_size = block_size
        self._buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._running = False
        self._stream = None
        self._use_sounddevice = _sounddevice_available()

        # File fallback state
        self._file_chunks: list[np.ndarray] = []

        # Stats
        self._frames_played = 0
        self._underruns = 0

    @property
    def frames_played(self) -> int:
        return self._frames_played

    @property
    def underruns(self) -> int:
        return self._underruns

    def enqueue(self, audio: np.ndarray) -> None:
        """Add an audio frame to the playback buffer.

        Parameters
        ----------
        audio : np.ndarray
            1-D float64 mono audio chunk.
        """
        audio = np.asarray(audio, dtype=np.float64).ravel()
        with self._lock:
            self._buffer.append(audio)

    def start(self) -> None:
        """Start the audio output stream."""
        if self._running:
            return
        self._running = True

        if self._use_sounddevice:
            import sounddevice as sd
            self._stream = sd.OutputStream(
                samplerate=self._sample_rate,
                channels=self._channels,
                blocksize=self._block_size,
                dtype="float32",
                callback=self._audio_callback,
            )
            self._stream.start()
        else:
            # File fallback: just collect chunks
            self._file_chunks = []

    def stop(self) -> None:
        """Stop the audio output stream."""
        self._running = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def save_to_file(self, path: str | Path) -> None:
        """Save all buffered audio to a WAV file (fallback mode).

        Also works as a post-hoc recording mechanism even in sounddevice mode.
        """
        from scipy.io import wavfile

        all_audio = list(self._file_chunks) if self._file_chunks else []
        with self._lock:
            all_audio.extend(self._buffer)

        if not all_audio:
            return

        combined = np.concatenate(all_audio)
        # Scale to int16 for WAV
        pcm = (combined * 32767).clip(-32768, 32767).astype(np.int16)
        wavfile.write(str(path), self._sample_rate, pcm)

    def _audio_callback(
        self,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: object,
    ) -> None:
        """sounddevice callback: fill output buffer from our queue."""
        if not self._running:
            outdata[:] = 0
            return

        filled = 0
        while filled < frames:
            with self._lock:
                if not self._buffer:
                    break
                chunk = self._buffer.popleft()

            remaining = frames - filled
            take = min(len(chunk), remaining)
            outdata[filled:filled + take, 0] = chunk[:take].astype(np.float32)
            filled += take

            if take < len(chunk):
                # Put back unused portion
                with self._lock:
                    self._buffer.appendleft(chunk[take:])

            self._frames_played += take

        if filled < frames:
            outdata[filled:] = 0
            self._underruns += 1

        # Also store for file saving
        self._file_chunks.append(outdata[:, 0].copy().astype(np.float64))

    def __enter__(self) -> "AudioPlayback":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.stop()
