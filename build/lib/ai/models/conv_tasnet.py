"""Conv-TasNet speech enhancement model for PS26052 ANC.

A from-scratch comparison architecture trained on the same Phase A dataset,
providing a clean from-scratch-vs-fine-tuned comparison alongside the
classical-vs-AI comparison.

Conv-TasNet (Convolutional Time-domain Audio Separation Network) operates
directly in the time domain using a learned encoder/decoder basis and
temporal convolutional network (TCN) for mask estimation.

Reference: Luo & Mesgarani (2019) "Conv-TasNet: Surpassing Ideal
Time-Frequency Magnitude Masking for Speech Separation"

This implementation is adapted for single-channel speech enhancement
(denoising) rather than separation.
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

# Default Conv-TasNet hyperparameters (sized for real-time on laptop CPU)
_DEFAULT_N = 256       # Number of encoder filters
_DEFAULT_L = 20        # Encoder filter length
_DEFAULT_B = 256       # Bottleneck channels
_DEFAULT_H = 512       # TCN hidden channels
_DEFAULT_P = 3         # TCN kernel size
_DEFAULT_X = 8         # Number of convolutional blocks in each repeat
_DEFAULT_R = 3         # Number of repeats
_SAMPLE_RATE = 16_000


if TORCH_AVAILABLE:

    class _Encoder(nn.Module):
        """1-D convolutional encoder (learned basis)."""

        def __init__(self, N: int = _DEFAULT_N, L: int = _DEFAULT_L) -> None:
            super().__init__()
            self.conv = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)
            self.relu = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: [batch, signal_length] -> [batch, N, T]"""
            x = x.unsqueeze(1)  # [batch, 1, signal_length]
            return self.relu(self.conv(x))

    class _Decoder(nn.Module):
        """1-D transposed convolutional decoder."""

        def __init__(self, N: int = _DEFAULT_N, L: int = _DEFAULT_L) -> None:
            super().__init__()
            self.deconv = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L // 2, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: [batch, N, T] -> [batch, signal_length]"""
            return self.deconv(x).squeeze(1)

    class _DepthwiseSeparableConv(nn.Module):
        """Depthwise separable convolution block for the TCN."""

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            kernel_size: int,
            dilation: int,
        ) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(in_channels, hidden_channels, 1),
                nn.PReLU(),
                nn.GroupNorm(1, hidden_channels),
                nn.Conv1d(
                    hidden_channels, hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=(kernel_size - 1) * dilation // 2,
                    groups=hidden_channels,
                ),
                nn.PReLU(),
                nn.GroupNorm(1, hidden_channels),
                nn.Conv1d(hidden_channels, in_channels, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.net(x)  # Residual connection

    class _TCN(nn.Module):
        """Temporal Convolutional Network for mask estimation."""

        def __init__(
            self,
            N: int = _DEFAULT_N,
            B: int = _DEFAULT_B,
            H: int = _DEFAULT_H,
            P: int = _DEFAULT_P,
            X: int = _DEFAULT_X,
            R: int = _DEFAULT_R,
        ) -> None:
            super().__init__()

            self.layer_norm = nn.GroupNorm(1, N)
            self.bottleneck = nn.Conv1d(N, B, 1)

            blocks = []
            for r in range(R):
                for x in range(X):
                    dilation = 2 ** x
                    blocks.append(_DepthwiseSeparableConv(B, H, P, dilation))

            self.tcn = nn.Sequential(*blocks)
            self.mask_conv = nn.Conv1d(B, N, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """x: [batch, N, T] -> mask: [batch, N, T]"""
            normed = self.layer_norm(x)
            bottleneck = self.bottleneck(normed)
            tcn_out = self.tcn(bottleneck)
            mask = self.sigmoid(self.mask_conv(tcn_out))
            return mask

    class ConvTasNet(nn.Module):
        """Conv-TasNet for single-channel speech enhancement.

        Encoder → TCN mask estimation → masked encoder output → Decoder.
        """

        def __init__(
            self,
            N: int = _DEFAULT_N,
            L: int = _DEFAULT_L,
            B: int = _DEFAULT_B,
            H: int = _DEFAULT_H,
            P: int = _DEFAULT_P,
            X: int = _DEFAULT_X,
            R: int = _DEFAULT_R,
        ) -> None:
            super().__init__()
            self.encoder = _Encoder(N, L)
            self.tcn = _TCN(N, B, H, P, X, R)
            self.decoder = _Decoder(N, L)
            self.L = L

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
            original_length = x.shape[-1]

            # Pad to multiple of stride
            stride = self.L // 2
            pad_needed = (stride - (original_length % stride)) % stride
            if pad_needed > 0:
                x = F.pad(x, (0, pad_needed))

            # Encode → Mask → Decode
            encoded = self.encoder(x)
            mask = self.tcn(encoded)
            masked = encoded * mask
            output = self.decoder(masked)

            # Trim to original length
            return output[:, :original_length]


class ConvTasNetModel(EnhancementModel):
    """Wraps a trained ConvTasNet for the EnhancementModel interface."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        target_sample_rate: int = 16_000,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ConvTasNetModel.")

        self._target_rate = target_sample_rate
        self._model: Any = None
        self._checkpoint_path = checkpoint_path

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        self._model = ConvTasNet()

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
        return "conv_tasnet"

    @property
    def sample_rate(self) -> int:
        return self._target_rate

    @property
    def frame_size(self) -> int:
        return 0  # Can handle arbitrary lengths

    def enhance(self, x: np.ndarray) -> np.ndarray:
        """Enhance mono audio through the Conv-TasNet model."""
        x = self.validate_input(x)
        self._ensure_model()
        original_length = len(x)

        # Resample if needed
        from anc.speech.resampling import resample_audio
        if self._target_rate != _SAMPLE_RATE:
            x_proc = resample_audio(x, self._target_rate, _SAMPLE_RATE)
        else:
            x_proc = x.copy()

        with torch.no_grad():
            x_tensor = torch.from_numpy(x_proc).float().unsqueeze(0)
            output_tensor = self._model(x_tensor)
            output = output_tensor.squeeze(0).numpy().astype(np.float64)

        # Resample back
        if self._target_rate != _SAMPLE_RATE:
            output = resample_audio(output, _SAMPLE_RATE, self._target_rate)

        # Match original length
        if len(output) > original_length:
            output = output[:original_length]
        elif len(output) < original_length:
            output = np.pad(output, (0, original_length - len(output)))

        return output
