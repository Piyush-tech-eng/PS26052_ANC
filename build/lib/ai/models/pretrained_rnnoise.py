"""Pretrained RNNoise wrapper implementing the EnhancementModel interface.

RNNoise is a lightweight recurrent neural network for real-time noise
suppression, operating on 48 kHz mono audio in 10 ms frames (480 samples).
This wrapper handles:

* Resampling from/to the pipeline's canonical rate using
  ``anc.speech.resampling``.
* Frame-based processing (480-sample chunks at 48 kHz).
* Graceful degradation: if RNNoise is unavailable, ``is_available()``
  returns ``False`` and the caller can fall back to another model.

Installation
------------
The ``rnnoise-python`` package provides Python bindings::

    pip install rnnoise-python

Alternatively, a compiled ``librnnoise`` shared library can be used via
ctypes.  Set the ``RNNOISE_LIB_PATH`` environment variable to point to it.
"""

from __future__ import annotations

import os
import warnings
from typing import Any

import numpy as np

from ai.models.base import EnhancementModel

# RNNoise native parameters
_RNNOISE_SAMPLE_RATE = 48_000
_RNNOISE_FRAME_SIZE = 480  # 10 ms at 48 kHz


def is_available() -> bool:
    """Check whether an RNNoise backend is importable."""
    try:
        import rnnoise  # noqa: F401
        return True
    except ImportError:
        pass
    # Fallback: check for a shared library via env var
    lib_path = os.environ.get("RNNOISE_LIB_PATH", "")
    if lib_path and os.path.isfile(lib_path):
        return True
    return False


def _get_rnnoise_backend() -> Any:
    """Return an RNNoise denoiser instance, or raise ImportError."""
    try:
        import rnnoise
        return rnnoise.RNNoise()
    except ImportError:
        pass

    lib_path = os.environ.get("RNNOISE_LIB_PATH", "")
    if lib_path and os.path.isfile(lib_path):
        import ctypes
        lib = ctypes.CDLL(lib_path)
        # Minimal ctypes wrapper — production code would be more robust
        return _CtypesRNNoise(lib)

    raise ImportError(
        "RNNoise is not available. Install rnnoise-python or set "
        "RNNOISE_LIB_PATH to a compiled librnnoise shared library."
    )


class _CtypesRNNoise:
    """Minimal ctypes wrapper around librnnoise for fallback use."""

    def __init__(self, lib: Any) -> None:
        self._lib = lib
        self._lib.rnnoise_create.restype = ctypes.c_void_p  # type: ignore[name-defined]
        self._lib.rnnoise_create.argtypes = [ctypes.c_void_p]  # type: ignore[name-defined]
        self._state = self._lib.rnnoise_create(None)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process one 480-sample frame through RNNoise."""
        import ctypes
        buf = (ctypes.c_float * _RNNOISE_FRAME_SIZE)()
        # RNNoise expects int16-range floats
        scaled = np.clip(frame * 32768.0, -32768.0, 32767.0).astype(np.float32)
        for i in range(_RNNOISE_FRAME_SIZE):
            buf[i] = float(scaled[i])
        self._lib.rnnoise_process_frame(self._state, buf, buf)
        out = np.array([buf[i] for i in range(_RNNOISE_FRAME_SIZE)], dtype=np.float64)
        return out / 32768.0

    def __del__(self) -> None:
        if hasattr(self, "_state") and self._state:
            try:
                self._lib.rnnoise_destroy(self._state)
            except Exception:
                pass


class RNNoiseModel(EnhancementModel):
    """Pretrained RNNoise wrapper for real-time speech enhancement.

    Parameters
    ----------
    target_sample_rate : int
        The pipeline's canonical sample rate.  Audio is resampled to 48 kHz
        for RNNoise processing and back to this rate on output.
    """

    def __init__(self, target_sample_rate: int = 16_000) -> None:
        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive.")
        self._target_rate = target_sample_rate
        self._backend: Any = None

    def _ensure_backend(self) -> None:
        """Lazy-init the RNNoise backend on first use."""
        if self._backend is None:
            self._backend = _get_rnnoise_backend()

    # ------------------------------------------------------------------
    # EnhancementModel interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "rnnoise"

    @property
    def sample_rate(self) -> int:
        return self._target_rate

    @property
    def frame_size(self) -> int:
        # Each RNNoise frame is 480 samples at 48 kHz.
        # Convert to the target rate for the streaming layer.
        from math import gcd
        ratio_num = self._target_rate
        ratio_den = _RNNOISE_SAMPLE_RATE
        g = gcd(ratio_num, ratio_den)
        return _RNNOISE_FRAME_SIZE * (ratio_num // g) // (ratio_den // g)

    def enhance(self, x: np.ndarray) -> np.ndarray:
        """Enhance mono audio through RNNoise.

        Resamples to 48 kHz, processes in 480-sample frames, resamples back.
        """
        x = self.validate_input(x)
        self._ensure_backend()
        original_length = len(x)

        # Resample to 48 kHz
        from anc.speech.resampling import resample_audio
        if self._target_rate != _RNNOISE_SAMPLE_RATE:
            x_48k = resample_audio(x, self._target_rate, _RNNOISE_SAMPLE_RATE)
        else:
            x_48k = x.copy()

        # Pad to a multiple of frame size
        num_48k = len(x_48k)
        pad_needed = (_RNNOISE_FRAME_SIZE - (num_48k % _RNNOISE_FRAME_SIZE)) % _RNNOISE_FRAME_SIZE
        if pad_needed > 0:
            x_48k = np.pad(x_48k, (0, pad_needed))

        # Process frame by frame
        num_frames = len(x_48k) // _RNNOISE_FRAME_SIZE
        output_48k = np.empty_like(x_48k)

        for i in range(num_frames):
            start = i * _RNNOISE_FRAME_SIZE
            end = start + _RNNOISE_FRAME_SIZE
            frame = x_48k[start:end]

            if hasattr(self._backend, "process_frame"):
                output_48k[start:end] = self._backend.process_frame(frame)
            else:
                # rnnoise-python API
                output_48k[start:end] = self._backend(frame)

        # Trim padding
        output_48k = output_48k[:num_48k]

        # Resample back to target rate
        if self._target_rate != _RNNOISE_SAMPLE_RATE:
            output = resample_audio(output_48k, _RNNOISE_SAMPLE_RATE, self._target_rate)
        else:
            output = output_48k

        # Match original length exactly
        if len(output) > original_length:
            output = output[:original_length]
        elif len(output) < original_length:
            output = np.pad(output, (0, original_length - len(output)))

        return output
