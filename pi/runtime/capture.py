"""Audio capture engine for Raspberry Pi 3 + ReSpeaker 2-Mic HAT."""

from __future__ import annotations

import sys
import numpy as np

from pi.runtime.config import PiConfig


class PiAudioCapture:
    """Manages low-latency two-channel audio capture via ALSA / PyAudio."""

    def __init__(self, config: PiConfig) -> None:
        self.config = config
        self._pa = None
        self._stream = None
        self._running = False

    def start(self) -> None:
        """Initialize PyAudio and start audio capture stream."""
        if self._running:
            return

        try:
            import pyaudio
        except ImportError:
            raise ImportError(
                "pyaudio is required for Pi audio capture. "
                "Install with: sudo apt-get install python3-pyaudio"
            )

        self._pa = pyaudio.PyAudio()
        dev_idx = self.config.device_index

        if dev_idx is None:
            for i in range(self._pa.get_device_count()):
                try:
                    info = self._pa.get_device_info_by_index(i)
                    name = info.get("name", "").lower()
                    if "respeaker" in name or "seeed" in name:
                        dev_idx = i
                        break
                except Exception:
                    pass

        stream_kwargs = {
            "format": pyaudio.paInt16,
            "channels": self.config.channels,
            "rate": self.config.sample_rate,
            "input": True,
            "frames_per_buffer": self.config.samples_per_frame,
        }
        if dev_idx is not None:
            stream_kwargs["input_device_index"] = dev_idx

        self._stream = self._pa.open(**stream_kwargs)
        self._running = True

    def read_frame(self) -> np.ndarray | None:
        """Read a single two-channel audio frame as int16 PCM."""
        if not self._running or self._stream is None:
            return None

        try:
            data = self._stream.read(
                self.config.samples_per_frame,
                exception_on_overflow=False,
            )
            return np.frombuffer(data, dtype=np.int16)
        except Exception:
            return None

    def stop(self) -> None:
        """Stop audio stream and release resources."""
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
