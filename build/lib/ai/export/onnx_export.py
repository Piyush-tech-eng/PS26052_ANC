"""Export trained PyTorch models to ONNX with numerical equivalence validation.

Exports the fine-tuned DTLN and Conv-TasNet models to ONNX format,
validating that the exported model produces output matching the source
PyTorch model within a strict tolerance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def export_to_onnx(
    model: Any,
    output_path: str | Path,
    *,
    signal_length: int = 16000,
    tolerance: float = 1e-4,
    opset_version: int = 17,
) -> dict[str, Any]:
    """Export a PyTorch model to ONNX with numerical validation.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model to export (must accept [batch, signal_length]).
    output_path : str or Path
        Path for the output .onnx file.
    signal_length : int
        Signal length for the example input.
    tolerance : float
        Maximum absolute error tolerance for validation.
    opset_version : int
        ONNX opset version.

    Returns
    -------
    dict
        Export results including max absolute error.

    Raises
    ------
    ValueError
        If numerical validation fails (max error exceeds tolerance).
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for ONNX export.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Create example input
    dummy_input = torch.randn(1, signal_length)

    # Get PyTorch reference output
    with torch.no_grad():
        pytorch_output = model(dummy_input).numpy()

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        str(output_file),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "signal_length"},
            "output": {0: "batch_size", 1: "signal_length"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )

    # Validate numerical equivalence
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠ onnxruntime not available — skipping numerical validation.")
        return {
            "output_path": str(output_file),
            "validated": False,
            "message": "onnxruntime not available for validation",
        }

    session = ort.InferenceSession(str(output_file))
    onnx_output = session.run(None, {"input": dummy_input.numpy()})[0]

    max_error = float(np.max(np.abs(pytorch_output - onnx_output)))

    result = {
        "output_path": str(output_file),
        "max_absolute_error": max_error,
        "tolerance": tolerance,
        "validated": max_error < tolerance,
    }

    if max_error >= tolerance:
        raise ValueError(
            f"ONNX export validation failed: max absolute error {max_error:.6f} "
            f"exceeds tolerance {tolerance:.6f}."
        )

    print(f"  ✓ ONNX export validated: max error = {max_error:.2e} < {tolerance:.2e}")
    return result
