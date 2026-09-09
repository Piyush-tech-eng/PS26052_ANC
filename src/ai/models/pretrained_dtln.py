"""Pretrained DTLN (Dual-signal Transformation LSTM Network) wrapper.

DTLN is a lightweight dual-stage speech enhancement architecture:
- Stage 1: LSTM in the STFT magnitude domain (257-bin mask estimation)
- Stage 2: LSTM in the time domain (learned basis frame enhancement)

Public pretrained weights are available from the original authors:
    https://github.com/breizhn/DTLN

This wrapper loads pretrained DTLN ONNX checkpoints (``model_1.onnx`` and
``model_2.onnx``) and adapts them to the ``EnhancementModel`` interface.
The model operates on 16 kHz mono audio with a block length of 512 samples
and a block shift of 128 samples.

Model I/O (from the pretrained ONNX files)
-------------------------------------------
model_1.onnx — STFT magnitude domain:
    Inputs:  input_2 [1,1,257]  (magnitude spectrum)
             input_3 [1,2,128,2] (packed LSTM states)
    Outputs: activation_2 [1,1,257]  (magnitude mask)
             tf_op_layer_stack_2 [1,2,128,2] (updated states)

model_2.onnx — time domain:
    Inputs:  input_4 [1,1,512]  (estimated time-domain frame)
             input_5 [1,2,128,2] (packed LSTM states)
    Outputs: conv1d_3 [1,1,512]  (enhanced time-domain frame)
             tf_op_layer_stack_5 [1,2,128,2] (updated states)

Graceful degradation
--------------------
If ONNX Runtime is not installed, ``is_available()`` returns ``False``
and the caller can fall back to another model.
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
_DTLN_FFT_SIZE = 512
_DTLN_NUM_BINS = _DTLN_FFT_SIZE // 2 + 1  # 257


def is_available() -> bool:
    """Check if a DTLN runtime (ONNX Runtime) is available."""
    try:
        import onnxruntime  # noqa: F401
        return True
    except ImportError:
        return False


def _find_model_path() -> Path | None:
    """Search standard locations for a pretrained DTLN checkpoint directory.

    Returns the first existing directory that contains at least one
    ``.onnx`` file, or ``None`` if nothing is found.
    """
    # Check environment variable first
    env_path = os.environ.get("DTLN_MODEL_PATH", "")
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    # Resolve repo root — walk up from this file until we hit a directory
    # that contains pyproject.toml (works regardless of CWD).
    _this_dir = Path(__file__).resolve().parent
    repo_root: Path | None = None
    for parent in [_this_dir, *_this_dir.parents]:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break

    # Check common relative-to-repo paths and then user cache
    candidates: list[Path] = []
    if repo_root is not None:
        candidates.append(repo_root / "models" / "dtln")
        candidates.append(repo_root / "pretrained" / "dtln")
    # CWD-relative (backward compat)
    candidates.append(Path("models/dtln"))
    candidates.append(Path("pretrained/dtln"))
    candidates.append(Path.home() / ".cache" / "dtln")

    seen: set[str] = set()
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.exists():
            # Verify it actually contains ONNX files
            if candidate.is_dir() and list(candidate.glob("*.onnx")):
                return candidate
            elif candidate.is_file() and candidate.suffix == ".onnx":
                return candidate
    return None


class DTLNModel(EnhancementModel):
    """Pretrained DTLN speech enhancement model.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to the pretrained DTLN model directory containing
        ``model_1.onnx`` and ``model_2.onnx``, or a single ONNX file.
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
        self._session: Any = None  # Lazy-loaded ONNX session dict

    def _ensure_session(self) -> None:
        """Lazy-load the ONNX sessions on first use."""
        if self._session is not None:
            return

        if self._model_path is None:
            raise FileNotFoundError(
                "No DTLN model found. Set DTLN_MODEL_PATH or place "
                "model_1.onnx + model_2.onnx in models/dtln/."
            )

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for DTLN inference. "
                "Install with: pip install onnxruntime"
            ) from exc

        # Locate ONNX files
        if self._model_path.is_dir():
            onnx_files = sorted(self._model_path.glob("*.onnx"))
        elif self._model_path.suffix == ".onnx":
            onnx_files = [self._model_path]
        else:
            raise FileNotFoundError(
                f"No .onnx files found at {self._model_path}"
            )

        if len(onnx_files) < 2:
            raise FileNotFoundError(
                f"DTLN requires model_1.onnx and model_2.onnx, "
                f"found: {[f.name for f in onnx_files]}"
            )

        # Create inference sessions
        sessions = [ort.InferenceSession(str(f)) for f in onnx_files[:2]]

        # Pre-allocate input dicts by inspecting model metadata
        init_inputs = []
        for sess in sessions:
            inputs = {}
            for inp in sess.get_inputs():
                shape = [
                    dim if isinstance(dim, int) else 1
                    for dim in inp.shape
                ]
                inputs[inp.name] = np.zeros(shape, dtype=np.float32)
            init_inputs.append(inputs)

        self._session = {
            "type": "onnx",
            "sessions": sessions,
            "init_inputs": init_inputs,
        }

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

        # DTLN overlap-add introduces a 384-sample (24 ms) lookback latency.
        # Pad with 384 tail samples so the full input is processed through the lookback,
        # and slice output from 384 to achieve sample-for-sample alignment.
        delay = _DTLN_BLOCK_LEN - _DTLN_BLOCK_SHIFT  # 384 samples
        num_16k = len(x_16k)
        x_padded = np.pad(x_16k, (0, delay + _DTLN_BLOCK_SHIFT))
        pad_needed = (_DTLN_BLOCK_SHIFT - (len(x_padded) % _DTLN_BLOCK_SHIFT)) % _DTLN_BLOCK_SHIFT
        if pad_needed:
            x_padded = np.pad(x_padded, (0, pad_needed))

        # Process through DTLN
        raw_output = self._process_onnx(x_padded)
        output_16k = raw_output[delay : delay + num_16k]

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
        """Process audio through the two-stage DTLN ONNX pipeline.

        Follows the reference implementation from breizhn/DTLN:
        - Stage 1 (model_1): STFT magnitude -> LSTM -> magnitude mask
        - Stage 2 (model_2): estimated time frame -> LSTM -> enhanced frame
        """
        sessions = self._session["sessions"]
        init_inputs = self._session["init_inputs"]

        sess_1, sess_2 = sessions[0], sessions[1]
        inp_names_1 = [inp.name for inp in sess_1.get_inputs()]
        inp_names_2 = [inp.name for inp in sess_2.get_inputs()]

        num_blocks = len(x) // _DTLN_BLOCK_SHIFT
        output = np.zeros(len(x), dtype=np.float32)

        # Sliding input buffer for overlap
        in_buffer = np.zeros(_DTLN_BLOCK_LEN, dtype=np.float32)
        out_buffer = np.zeros(_DTLN_BLOCK_LEN, dtype=np.float32)

        # LSTM states -- initialized from model metadata
        # model_1 state: [1, 2, 128, 2]  (packed LSTM h+c for 2 layers)
        # model_2 state: [1, 2, 128, 2]
        states_1 = init_inputs[0][inp_names_1[1]].copy()
        states_2 = init_inputs[1][inp_names_2[1]].copy()

        for i in range(num_blocks):
            start = i * _DTLN_BLOCK_SHIFT

            # Shift input buffer and append new block
            in_buffer[:-_DTLN_BLOCK_SHIFT] = in_buffer[_DTLN_BLOCK_SHIFT:]
            in_buffer[-_DTLN_BLOCK_SHIFT:] = x[start:start + _DTLN_BLOCK_SHIFT].astype(np.float32)

            # ---- Stage 1: STFT domain ----
            # Compute magnitude spectrum
            in_block_fft = np.fft.rfft(in_buffer)
            in_mag = np.abs(in_block_fft).astype(np.float32)
            in_phase = np.angle(in_block_fft)

            # Run model_1: magnitude -> mask
            mag_input = in_mag.reshape(1, 1, _DTLN_NUM_BINS)
            result_1 = sess_1.run(None, {
                inp_names_1[0]: mag_input,
                inp_names_1[1]: states_1,
            })
            out_mask = result_1[0]       # [1, 1, 257] mask
            states_1 = result_1[1]       # updated LSTM states

            # Apply mask in STFT domain: mask * in_mag * exp(1j * in_phase)
            estimated_complex = in_mag * out_mask.flatten() * np.exp(1j * in_phase)
            estimated_block = np.fft.irfft(estimated_complex).astype(np.float32)

            # ---- Stage 2: time domain ----
            est_input = estimated_block.reshape(1, 1, _DTLN_BLOCK_LEN)
            result_2 = sess_2.run(None, {
                inp_names_2[0]: est_input,
                inp_names_2[1]: states_2,
            })
            out_block = result_2[0].flatten()  # [512] enhanced frame
            states_2 = result_2[1]             # updated LSTM states

            # Overlap-add output
            out_buffer[:-_DTLN_BLOCK_SHIFT] = out_buffer[_DTLN_BLOCK_SHIFT:]
            out_buffer[-_DTLN_BLOCK_SHIFT:] = 0.0
            out_buffer += out_block[:_DTLN_BLOCK_LEN]
            output[start:start + _DTLN_BLOCK_SHIFT] = out_buffer[:_DTLN_BLOCK_SHIFT]

        return output.astype(np.float64)
