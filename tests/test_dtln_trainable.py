"""Tests for trainable DTLN and ONNX-to-PyTorch weight transfer."""

import numpy as np
import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai.models.pretrained_dtln import DTLNModel, is_available as onnx_available
from ai.models.dtln_trainable import (
    DTLNTrainableModel,
    TrainableDTLN,
    load_from_onnx,
    is_available as torch_available,
)


@pytest.mark.skipif(
    not (onnx_available() and torch_available()),
    reason="Both ONNX Runtime and PyTorch required for conversion test",
)
def test_onnx_trainable_numerical_equivalence():
    """Acceptance check: PyTorch model loaded from ONNX matches ONNX output within atol=1e-3."""
    from pathlib import Path

    model_dir = Path("models/dtln")
    m1_path = model_dir / "model_1.onnx"
    m2_path = model_dir / "model_2.onnx"
    if not (m1_path.exists() and m2_path.exists()):
        pytest.skip("Pretrained DTLN ONNX models not found in models/dtln")

    # Generate synthetic input signal
    np.random.seed(42)
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    x = (0.4 * np.sin(2 * np.pi * 300 * t) + 0.2 * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.random.randn(16000)).astype(np.float64)

    # 1. Pretrained ONNX inference
    onnx_model = DTLNModel("models/dtln")
    y_onnx = onnx_model.enhance(x)

    # 2. Trainable PyTorch loaded from ONNX
    torch_model = DTLNTrainableModel()  # auto-loads from models/dtln
    y_torch = torch_model.enhance(x)

    # Maximum difference and correlation checks
    max_diff = np.max(np.abs(y_onnx - y_torch))
    corr = np.corrcoef(y_onnx, y_torch)[0, 1]

    assert corr > 0.999, f"Correlation too low: {corr}"
    assert np.allclose(y_onnx, y_torch, atol=1e-3), f"Max diff exceeded atol=1e-3: {max_diff}"
    # Even stricter: our verified exact mapping gives < 1e-6 error
    assert max_diff < 1e-5, f"Expected near-exact numerical match, got {max_diff}"


@pytest.mark.skipif(not torch_available(), reason="PyTorch required")
def test_trainable_dtln_backward_pass():
    """Ensure all parameters of TrainableDTLN participate in gradient computation."""
    model = TrainableDTLN()
    model.train()

    # Input: batch of 2 audio signals, 1024 samples
    x = torch.randn(2, 1024, requires_grad=True)
    out = model(x)
    loss = out.pow(2).sum()
    loss.backward()

    # All parameters must receive non-zero gradients
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Parameter {name} did not receive a gradient."
        assert not torch.isnan(param.grad).any(), f"Parameter {name} gradient contains NaN."
        assert torch.count_nonzero(param.grad) > 0, f"Parameter {name} gradient is all zeros."


@pytest.mark.skipif(not torch_available(), reason="PyTorch required")
def test_dtln_trainable_model_interface():
    """Verify DTLNTrainableModel complies with EnhancementModel interface."""
    model = DTLNTrainableModel()
    assert model.name == "dtln_finetuned"
    assert model.sample_rate == 16000
    assert model.frame_size == 128

    # Process audio
    x = np.zeros(16000, dtype=np.float64)
    y = model.enhance(x)
    assert len(y) == len(x)
    assert isinstance(y, np.ndarray)
