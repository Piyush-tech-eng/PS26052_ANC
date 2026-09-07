"""Pretrained DTLN (Dual-signal Transformation LSTM Network) wrapper.

DTLN is a lightweight dual-stage speech enhancement architecture:
- Stage 1: LSTM in the STFT magnitude domain
- Stage 2: LSTM in the time domain (learned basis)

Public pretrained weights are available from the original authors:
    https://github.com/breizhn/DTLN

This wrapper loads a pretrained DTLN checkpoint (TensorFlow SavedModel or
ONNX) and adapts it to the ``EnhancementModel`` interface.  The model
operates on 16 kHz mono audio with a block length of 512 samples and a
block shift of 128 samples.

Graceful degradation
--------------------
If neither TensorFlow nor ONNX Runtime is available, ``is_available()``
returns ``False`` and the caller can fall back to another model.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ai.models.base import EnhancementModel

# DTLN native parameters
_DTLN_SAMPLE_RATE = 16_000
_DTLN_BLOCK_LEN = 512
_DTLN_BLOCK_SHIFT = 128


def is_available() -> bool:
    """Check if a DTLN runtime is available."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        pass
    try:
        import tensorflow  # noqa: F401
        return True
    except ImportError:
        pass
    return False


def _find_model_path() -> Path | None:
    """Search standard locations for a pretrained DTLN checkpoint."""
    # Check environment variable first
    env_path = os.environ.get("DTLN_MODEL_PATH", "")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Check common relative paths
    candidates = [
        Path("models/dtln"),
        Path("pretrained/dtln"),
        Path.home() / ".cache" / "dtln",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return None


class DTLNModel(EnhancementModel):
    """Pretrained DTLN speech enhancement model.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the pretrained DTLN model directory or ONNX file.
        If not provided, searches standard locations and environment
        variable ``DTLN_MODEL_PATH``.
    target_sample_rate : int
        Pipeline's canonical sample rate.  If different from 16 kHz,
        audio is resampled transparently.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        target_sample_rate: int = 16_000,
    ) -> None:
        if target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive.")
        self._target_rate = target_sample_rate
        self._model_path = Path(model_path) if model_path else _find_model_path()
        self._session: Any = None  # Lazy-loaded ONNX or TF session

    def _ensure_session(self) -> None:
        """Lazy-load the model on first use."""
        if self._session is not None:
            return

        if self._model_path is None:
            raise FileNotFoundError(
                "No DTLN model found. Set DTLN_MODEL_PATH or pass model_path."
            )

        # Try ONNX Runtime first (lighter weight)
        onnx_files = list(self._model_path.glob("*.onnx")) if self._model_path.is_dir() else []
        if self._model_path.suffix == ".onnx":
            onnx_files = [self._model_path]

        if onnx_files:
            try:
                import onnxruntime as ort
                self._session = {
                    "type": "onnx",
                    "sessions": [
                        ort.InferenceSession(str(f)) for f in sorted(onnx_files)[:2]
                    ],
                }
                return
            except ImportError:
                pass

        # Fall back to TensorFlow SavedModel
        try:
            import tensorflow as tf
            self._session = {
                "type": "tensorflow",
                "model": tf.saved_model.load(str(self._model_path)),
            }
            return
        except (ImportError, Exception) as exc:
            raise ImportError(
                f"Cannot load DTLN model from {self._model_path}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # EnhancementModel interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "dtln"

    @property
    def sample_rate(self) -> int:
        return self._target_rate

    @property
    def frame_size(self) -> int:
        # Return block_shift in terms of target sample rate
        if self._target_rate == _DTLN_SAMPLE_RATE:
            return _DTLN_BLOCK_SHIFT
        return int(_DTLN_BLOCK_SHIFT * self._target_rate / _DTLN_SAMPLE_RATE)

    def enhance(self, x: np.ndarray) -> np.ndarray:
        """Enhance mono audio through the pretrained DTLN model.

        Resamples to 16 kHz if needed, processes block-by-block, resamples back.
        """
        x = self.validate_input(x)
        self._ensure_session()
        original_length = len(x)

        # Resample to 16 kHz if needed
        from anc.speech.resampling import resample_audio
        if self._target_rate != _DTLN_SAMPLE_RATE:
            x_16k = resample_audio(x, self._target_rate, _DTLN_SAMPLE_RATE)
        else:
            x_16k = x.copy()

        # Pad to multiple of block_shift
        num_16k = len(x_16k)
        pad_needed = (_DTLN_BLOCK_SHIFT - (num_16k % _DTLN_BLOCK_SHIFT)) % _DTLN_BLOCK_SHIFT
        if pad_needed:
            x_16k = np.pad(x_16k, (0, pad_needed))

        # Process through DTLN
        if self._session["type"] == "onnx":
            output_16k = self._process_onnx(x_16k)
        else:
            output_16k = self._process_tensorflow(x_16k)

        output_16k = output_16k[:num_16k]

        # Resample back
        if self._target_rate != _DTLN_SAMPLE_RATE:
            output = resample_audio(output_16k, _DTLN_SAMPLE_RATE, self._target_rate)
        else:
            output = output_16k

        # Match original length
        if len(output) > original_length:
            output = output[:original_length]
        elif len(output) < original_length:
            output = np.pad(output, (0, original_length - len(output)))

        return output

    def _process_onnx(self, x: np.ndarray) -> np.ndarray:
        """Process audio through ONNX Runtime sessions."""
        import onnxruntime as ort  # noqa: F811

        sessions = self._session["sessions"]
        num_blocks = len(x) // _DTLN_BLOCK_SHIFT
        output = np.zeros_like(x)

        # Initialize hidden states
        in_buffer = np.zeros(_DTLN_BLOCK_LEN, dtype=np.float32)
        out_buffer = np.zeros(_DTLN_BLOCK_LEN, dtype=np.float32)

        # DTLN typically uses 2 ONNX models (2 stages)
        h1 = np.zeros((1, 1, 128), dtype=np.float32)
        c1 = np.zeros((1, 1, 128), dtype=np.float32)
        h2 = np.zeros((1, 1, 128), dtype=np.float32)
        c2 = np.zeros((1, 1, 128), dtype=np.float32)

        for i in range(num_blocks):
            start = i * _DTLN_BLOCK_SHIFT
            # Shift buffer
            in_buffer[:-_DTLN_BLOCK_SHIFT] = in_buffer[_DTLN_BLOCK_SHIFT:]
            in_buffer[-_DTLN_BLOCK_SHIFT:] = x[start:start + _DTLN_BLOCK_SHIFT].astype(np.float32)

            if len(sessions) >= 2:
                # Stage 1: STFT domain
                inp = in_buffer.reshape(1, 1, -1)
                result1 = sessions[0].run(None, {
                    sessions[0].get_inputs()[0].name: inp,
                    sessions[0].get_inputs()[1].name: h1,
                    sessions[0].get_inputs()[2].name: c1,
                })
                estimated_block = result1[0].flatten()
                h1, c1 = result1[1], result1[2]

                # Stage 2: time domain
                inp2 = estimated_block.reshape(1, 1, -1)
                result2 = sessions[1].run(None, {
                    sessions[1].get_inputs()[0].name: inp2,
                    sessions[1].get_inputs()[1].name: h2,
                    sessions[1].get_inputs()[2].name: c2,
                })
                out_block = result2[0].flatten()
                h2, c2 = result2[1], result2[2]
            else:
                # Single combined model
                inp = in_buffer.reshape(1, 1, -1)
                result = sessions[0].run(None, {
                    sessions[0].get_inputs()[0].name: inp,
                })
                out_block = result[0].flatten()

            # Overlap-add output
            out_buffer[:-_DTLN_BLOCK_SHIFT] = out_buffer[_DTLN_BLOCK_SHIFT:]
            out_buffer[-_DTLN_BLOCK_SHIFT:] = 0.0
            out_buffer += out_block[:_DTLN_BLOCK_LEN]
            output[start:start + _DTLN_BLOCK_SHIFT] = out_buffer[:_DTLN_BLOCK_SHIFT]

        return output.astype(np.float64)

    def _process_tensorflow(self, x: np.ndarray) -> np.ndarray:
        """Process audio through TensorFlow SavedModel."""
        import tensorflow as tf

        model = self._session["model"]
        x_tensor = tf.constant(x.reshape(1, -1).astype(np.float32))
        output = model(x_tensor)
        if isinstance(output, dict):
            output = list(output.values())[0]
        return output.numpy().flatten().astype(np.float64)
