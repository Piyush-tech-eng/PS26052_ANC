"""Lightweight on-Pi neural enhancement inference wrapper."""

from __future__ import annotations

from pathlib import Path
import numpy as np


class LocalPiAI:
    """Runs quantized INT8 or FP32 ONNX DTLN directly on Raspberry Pi 3.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing model_1.onnx and model_2.onnx.
    sample_rate : int
        Sample rate (default 16000).
    """

    def __init__(self, model_dir: str | Path = "models/dtln_quantized", sample_rate: int = 16_000) -> None:
        self.model_dir = Path(model_dir)
        self.sample_rate = sample_rate
        self.session_1 = None
        self.session_2 = None
        self.frame_len = 512
        self.hop_size = 128
        self._inp_names_1: list[str] = []
        self._inp_names_2: list[str] = []
        self._states_1: np.ndarray | None = None
        self._states_2: np.ndarray | None = None

        self._in_buffer = np.zeros(self.frame_len, dtype=np.float32)
        self._out_buffer = np.zeros(self.frame_len, dtype=np.float32)

    def load(self) -> bool:
        """Load ONNX sessions with optimal single-thread CPU settings."""
        try:
            import onnxruntime as ort
        except ImportError:
            return False

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        m1 = self.model_dir / "model_1.onnx"
        m2 = self.model_dir / "model_2.onnx"

        if not m1.exists() or not m2.exists():
            return False

        self.session_1 = ort.InferenceSession(str(m1), sess_options=opts, providers=["CPUExecutionProvider"])
        self.session_2 = ort.InferenceSession(str(m2), sess_options=opts, providers=["CPUExecutionProvider"])

        self._inp_names_1 = [inp.name for inp in self.session_1.get_inputs()]
        self._inp_names_2 = [inp.name for inp in self.session_2.get_inputs()]

        # Initialize state tensors from shapes
        state1_shape = [dim if isinstance(dim, int) else 1 for dim in self.session_1.get_inputs()[1].shape]
        state2_shape = [dim if isinstance(dim, int) else 1 for dim in self.session_2.get_inputs()[1].shape]
        self._states_1 = np.zeros(state1_shape, dtype=np.float32)
        self._states_2 = np.zeros(state2_shape, dtype=np.float32)
        return True

    def process_frame(self, chunk: np.ndarray) -> np.ndarray:
        """Process one chunk of 128 or 320 samples."""
        if self.session_1 is None or self.session_2 is None:
            return chunk.copy()

        chunk_f32 = np.asarray(chunk, dtype=np.float32).ravel()
        n = len(chunk_f32)
        out = np.zeros(n, dtype=np.float32)

        # Process in hop_size increments
        for i in range(0, n, self.hop_size):
            sub = chunk_f32[i:i + self.hop_size]
            sub_len = len(sub)
            self._in_buffer[:-sub_len] = self._in_buffer[sub_len:]
            self._in_buffer[-sub_len:] = sub

            # Stage 1: STFT
            in_block_fft = np.fft.rfft(self._in_buffer)
            in_mag = np.abs(in_block_fft).astype(np.float32)
            in_phase = np.angle(in_block_fft)

            mag_input = in_mag.reshape(1, 1, 257)
            res1 = self.session_1.run(None, {
                self._inp_names_1[0]: mag_input,
                self._inp_names_1[1]: self._states_1,
            })
            mask = res1[0]
            self._states_1 = res1[1]

            estimated_complex = in_mag * mask.flatten() * np.exp(1j * in_phase)
            estimated_block = np.fft.irfft(estimated_complex).astype(np.float32)

            # Stage 2: Time domain
            est_input = estimated_block.reshape(1, 1, self.frame_len)
            res2 = self.session_2.run(None, {
                self._inp_names_2[0]: est_input,
                self._inp_names_2[1]: self._states_2,
            })
            out_block = res2[0].flatten()
            self._states_2 = res2[1]

            self._out_buffer[:-self.hop_size] = self._out_buffer[self.hop_size:]
            self._out_buffer[-self.hop_size:] = 0.0
            self._out_buffer += out_block[:self.frame_len]

            out[i:i + sub_len] = self._out_buffer[:sub_len]

        return out.astype(np.float64)
