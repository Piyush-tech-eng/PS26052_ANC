"""Real-time streaming infrastructure for ANC + AI enhancement.

This package provides:
- ``overlap_add``: Frame-based overlap-add processor for any EnhancementModel
- ``frame_anc``: Stateful frame-by-frame FxNLMS adaptation
- ``hybrid_engine``: Cascaded ANC → AI pipeline with latency instrumentation
"""

from ai.streaming.overlap_add import OverlapAddProcessor
from ai.streaming.frame_anc import FrameANC
from ai.streaming.hybrid_engine import HybridEngine

__all__ = ["OverlapAddProcessor", "FrameANC", "HybridEngine"]
