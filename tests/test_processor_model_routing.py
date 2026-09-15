"""
Test model routing paths through AudioProcessingPipeline.

Verifies that processor.py correctly dispatches to the rnnoise and conv_tasnet
model backends (which exist identically on both main and dashboard branches).
Relocated from dashboard's root-level test_processor.py and converted to proper
pytest assertions.
"""

import sys
import os
import numpy as np
import pytest

# Ensure src/ is on the import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from anc.interface.processor import AudioProcessingPipeline


@pytest.fixture(scope="module")
def pipeline():
    """Shared AudioProcessingPipeline instance for all tests in this module."""
    return AudioProcessingPipeline()


@pytest.fixture(scope="module")
def dummy_audio():
    """2-second random audio at 16 kHz, float32."""
    rng = np.random.default_rng(42)
    return rng.standard_normal(16000 * 2).astype(np.float32)


def test_dtln_quantized_routing(pipeline, dummy_audio):
    """Verify dtln_quantized model routes through the pipeline without error."""
    result = pipeline.process(dummy_audio, sample_rate=16000, model_name="dtln_quantized")
    assert result.success is True, f"dtln_quantized processing failed: {result}"


def test_rnnoise_routing(pipeline, dummy_audio):
    """Verify rnnoise model routes through the pipeline without error."""
    try:
        result = pipeline.process(dummy_audio, sample_rate=16000, model_name="rnnoise")
    except ImportError as e:
        if "RNNoise is not available" in str(e):
            pytest.skip("rnnoise native library not installed (optional dependency)")
        raise
    assert result.success is True, f"rnnoise processing failed: {result}"


def test_conv_tasnet_routing(pipeline, dummy_audio):
    """Verify conv_tasnet model routes through the pipeline without error."""
    result = pipeline.process(dummy_audio, sample_rate=16000, model_name="conv_tasnet")
    assert result.success is True, f"conv_tasnet processing failed: {result}"
