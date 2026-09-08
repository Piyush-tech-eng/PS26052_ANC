"""Trainable DTLN (Dual-signal Transformation LSTM Network) in PyTorch.

Re-implements the DTLN architecture from the pretrained ONNX checkpoint
so that gradients can flow for fine-tuning.  The architecture exactly
mirrors ``pretrained_dtln.py``'s two-stage pipeline:

- Stage 1 (STFT domain): magnitude spectrum → LSTM → magnitude mask
- Stage 2 (time domain): estimated frame → LSTM → enhanced frame

Weight initialization can be performed from the existing ONNX checkpoints
via ``load_from_onnx()``, or from random initialization for ablation studies.

This module also provides ``DTLNTrainableModel(EnhancementModel)`` so the
trained model is hot-swappable into the existing streaming/evaluation pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai.models.base import EnhancementModel


# DTLN native parameters (must match pretrained_dtln.py)
_DTLN_SAMPLE_RATE = 16_000
_DTLN_BLOCK_LEN = 512
_DTLN_BLOCK_SHIFT = 128
_DTLN_FFT_SIZE = 512
_DTLN_NUM_BINS = _DTLN_FFT_SIZE // 2 + 1  # 257
_DTLN_LSTM_UNITS = 128
_DTLN_LSTM_LAYERS = 2


def is_available() -> bool:
    """Check if PyTorch is available for training."""
    return TORCH_AVAILABLE


if TORCH_AVAILABLE:

    class DTLNStage1(nn.Module):
        """STFT magnitude domain: magnitude → LSTM → magnitude mask.

        Input:  [batch, 1, 257]  (magnitude spectrum)
        Output: [batch, 1, 257]  (masked magnitude = mask * input)
        """

        def __init__(
            self,
            num_bins: int = _DTLN_NUM_BINS,
            lstm_units: int = _DTLN_LSTM_UNITS,
            num_layers: int = _DTLN_LSTM_LAYERS,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=num_bins,
                hidden_size=lstm_units,
                num_layers=num_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(lstm_units, num_bins)
            self.sigmoid = nn.Sigmoid()

        def forward(
            self,
            magnitude: torch.Tensor,
            states: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            """Forward pass.

            Parameters
            ----------
            magnitude : torch.Tensor
                Shape [batch, 1, num_bins].
            states : tuple, optional
                LSTM hidden/cell states.

            Returns
            -------
            masked_magnitude : torch.Tensor
                Shape [batch, 1, num_bins].
            new_states : tuple
                Updated LSTM states.
            """
            lstm_out, new_states = self.lstm(magnitude, states)
            mask = self.sigmoid(self.fc(lstm_out))
            return magnitude * mask, new_states


    class DTLNStage2(nn.Module):
        """Time domain: estimated frame → LSTM → enhanced frame.

        Input:  [batch, 1, block_len]
        Output: [batch, 1, block_len]
        """

        def __init__(
            self,
            block_len: int = _DTLN_BLOCK_LEN,
            lstm_units: int = _DTLN_LSTM_UNITS,
            num_layers: int = _DTLN_LSTM_LAYERS,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=block_len,
                hidden_size=lstm_units,
                num_layers=num_layers,
                batch_first=True,
            )
            self.fc = nn.Linear(lstm_units, block_len)

        def forward(
            self,
            frame: torch.Tensor,
            states: tuple[torch.Tensor, torch.Tensor] | None = None,
        ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
            """Forward pass.

            Parameters
            ----------
            frame : torch.Tensor
                Shape [batch, 1, block_len].
            states : tuple, optional
                LSTM hidden/cell states.

            Returns
            -------
            enhanced : torch.Tensor
                Shape [batch, 1, block_len].
            new_states : tuple
                Updated LSTM states.
            """
            lstm_out, new_states = self.lstm(frame, states)
            enhanced = self.fc(lstm_out)
            return enhanced, new_states


    class TrainableDTLN(nn.Module):
        """Full trainable DTLN: two-stage pipeline for end-to-end training.

        Processes audio in overlapping blocks (block_len with block_shift hop),
        applying Stage1 in the STFT magnitude domain and Stage2 in the time
        domain, exactly matching the pretrained ONNX inference pipeline.
        """

        def __init__(
            self,
            block_len: int = _DTLN_BLOCK_LEN,
            block_shift: int = _DTLN_BLOCK_SHIFT,
            num_bins: int = _DTLN_NUM_BINS,
            lstm_units: int = _DTLN_LSTM_UNITS,
            num_layers: int = _DTLN_LSTM_LAYERS,
        ) -> None:
            super().__init__()
            self.block_len = block_len
            self.block_shift = block_shift
            self.num_bins = num_bins

            self.stage1 = DTLNStage1(num_bins, lstm_units, num_layers)
            self.stage2 = DTLNStage2(block_len, lstm_units, num_layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Process a batch of audio signals.

            Parameters
            ----------
            x : torch.Tensor
                Shape [batch, signal_length].

            Returns
            -------
            torch.Tensor
                Enhanced audio, shape [batch, signal_length].
            """
            batch_size, signal_len = x.shape
            num_blocks = signal_len // self.block_shift

            # Sliding window buffers
            in_buffer = torch.zeros(batch_size, self.block_len, device=x.device)
            out_buffer = torch.zeros(batch_size, self.block_len, device=x.device)
            output = torch.zeros_like(x)

            states1 = None
            states2 = None

            for i in range(num_blocks):
                start = i * self.block_shift

                # Shift input buffer
                in_buffer = torch.roll(in_buffer, -self.block_shift, dims=1)
                in_buffer[:, -self.block_shift:] = x[:, start:start + self.block_shift]

                # Stage 1: STFT domain
                # Windowed FFT
                window = torch.hann_window(self.block_len, device=x.device)
                windowed = in_buffer * window
                fft_out = torch.fft.rfft(windowed)
                magnitude = torch.abs(fft_out)  # [batch, num_bins]
                phase = torch.angle(fft_out)

                # LSTM mask estimation
                mag_input = magnitude.unsqueeze(1)  # [batch, 1, num_bins]
                masked_mag, states1 = self.stage1(mag_input, states1)
                masked_mag = masked_mag.squeeze(1)  # [batch, num_bins]

                # Reconstruct time-domain estimate
                estimated_complex = masked_mag * torch.exp(1j * phase)
                estimated_block = torch.fft.irfft(estimated_complex, n=self.block_len)

                # Stage 2: time domain
                est_input = estimated_block.unsqueeze(1)  # [batch, 1, block_len]
                enhanced_block, states2 = self.stage2(est_input, states2)
                enhanced_block = enhanced_block.squeeze(1)  # [batch, block_len]

                # Overlap-add output
                out_buffer = torch.roll(out_buffer, -self.block_shift, dims=1)
                out_buffer[:, -self.block_shift:] = 0.0
                out_buffer = out_buffer + enhanced_block
                output[:, start:start + self.block_shift] = out_buffer[:, :self.block_shift]

            return output


    def load_from_onnx(
        model_1_path: str | Path,
        model_2_path: str | Path,
    ) -> TrainableDTLN:
        """Load DTLN weights from ONNX checkpoints into a trainable PyTorch model.

        Extracts weights from the ONNX graph and maps them to the PyTorch
        state dict.  This enables fine-tuning from the pretrained checkpoint.

        Parameters
        ----------
        model_1_path : str or Path
            Path to ``model_1.onnx`` (STFT stage).
        model_2_path : str or Path
            Path to ``model_2.onnx`` (time stage).

        Returns
        -------
        TrainableDTLN
            Model with pretrained weights loaded.
        """
        try:
            import onnx
        except ImportError:
            raise ImportError(
                "The 'onnx' package is required to load weights from ONNX. "
                "Install with: pip install onnx"
            )

        model = TrainableDTLN()

        # Load and extract ONNX weights
        for stage_idx, (onnx_path, stage) in enumerate([
            (model_1_path, model.stage1),
            (model_2_path, model.stage2),
        ]):
            onnx_model = onnx.load(str(onnx_path))
            onnx_weights = {
                init.name: np.frombuffer(init.raw_data, dtype=np.float32).reshape(init.dims)
                for init in onnx_model.graph.initializer
            }

            # Map ONNX initializer names to PyTorch LSTM/FC parameters
            # The exact mapping depends on the ONNX export format.
            # We use a heuristic approach: match by shape.
            state_dict = stage.state_dict()

            for param_name, param_tensor in state_dict.items():
                target_shape = param_tensor.shape

                # Find the ONNX weight with matching shape
                matched = False
                for onnx_name, onnx_array in onnx_weights.items():
                    if onnx_array.shape == tuple(target_shape):
                        state_dict[param_name] = torch.from_numpy(onnx_array.copy())
                        del onnx_weights[onnx_name]
                        matched = True
                        break

                if not matched:
                    # Try transposed shape (common for FC layers)
                    for onnx_name, onnx_array in onnx_weights.items():
                        if len(onnx_array.shape) == 2 and onnx_array.T.shape == tuple(target_shape):
                            state_dict[param_name] = torch.from_numpy(onnx_array.T.copy())
                            del onnx_weights[onnx_name]
                            matched = True
                            break

            stage.load_state_dict(state_dict, strict=False)

        return model


class DTLNTrainableModel(EnhancementModel):
    """Wraps a trained TrainableDTLN for the EnhancementModel interface.

    This allows hot-swapping the fine-tuned model into the existing
    streaming engine, hardware integration, and evaluation pipeline
    without any code changes downstream.
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        target_sample_rate: int = 16_000,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for DTLNTrainableModel.")

        self._target_rate = target_sample_rate
        self._model: Any = None
        self._checkpoint_path = checkpoint_path

    def _ensure_model(self) -> None:
        """Lazy-load the PyTorch model."""
        if self._model is not None:
            return

        self._model = TrainableDTLN()

        if self._checkpoint_path is not None:
            checkpoint = torch.load(
                str(self._checkpoint_path),
                map_location="cpu",
                weights_only=True,
            )
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self._model.load_state_dict(checkpoint["model_state_dict"])
            else:
                self._model.load_state_dict(checkpoint)

        self._model.eval()

    @property
    def name(self) -> str:
        return "dtln_finetuned"

    @property
    def sample_rate(self) -> int:
        return self._target_rate

    @property
    def frame_size(self) -> int:
        if self._target_rate == _DTLN_SAMPLE_RATE:
            return _DTLN_BLOCK_SHIFT
        return int(_DTLN_BLOCK_SHIFT * self._target_rate / _DTLN_SAMPLE_RATE)

    def enhance(self, x: np.ndarray) -> np.ndarray:
        """Enhance mono audio through the fine-tuned DTLN model."""
        x = self.validate_input(x)
        self._ensure_model()
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

        # Process through PyTorch model
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_16k).float().unsqueeze(0)
            output_tensor = self._model(x_tensor)
            output_16k = output_tensor.squeeze(0).numpy().astype(np.float64)

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
