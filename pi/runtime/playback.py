"""ALSA playback sink for Raspberry Pi 3 audio output."""

from __future__ import annotations

import numpy as np


class PiAudioPlayback:
    """Manages low-latency audio playback on Pi via 3.5mm jack or USB DAC."""

    def __init__(self, sample_rate: int = 16_000, channels: int = 1) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._pa = None
        self._stream = None
        self._running = False

    def start(self) -> None:
        if self._running:
            return

        try:
            import pyaudio
        except ImportError:
            raise ImportError("pyaudio is required for Pi audio playback.")

        self._pa = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sample_rate,
            output=True,
            frames_per_buffer=320,
        )
        self._running = True

    def write_frame(self, audio_data: np.ndarray) -> None:
        """Write float64 [-1.0, 1.0] audio to DAC."""
        if not self._running or self._stream is None:
            return

        pcm = (audio_data * 32767).clip(-32768, 32767).astype(np.int16)
        try:
            self._stream.write(pcm.tobytes(), exception_on_underflow=False)
        except Exception:
            pass

    def stop(self) -> None:
        self._running = False
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

        if self._pa is not None:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
