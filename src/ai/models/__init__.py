"""Pluggable speech enhancement model backends.

All models implement :class:`ai.models.base.EnhancementModel` so the
streaming engine and hardware integration code depend only on the interface,
never on a specific model implementation.

Use :func:`get_best_available_model` to get the best model that can actually
run on this machine — it walks the fallback chain DTLN → RNNoise →
SpectralGate and returns the first one that loads successfully.
"""

from __future__ import annotations

import warnings
from typing import Literal

from ai.models.base import EnhancementModel


def get_best_available_model(
    *,
    prefer: Literal["auto", "dtln", "rnnoise", "spectral_gate"] = "auto",
    sample_rate: int = 16_000,
) -> EnhancementModel:
    """Return the best available enhancement model.

    Parameters
    ----------
    prefer : str
        Which model to try first.  ``"auto"`` walks the full fallback
        chain (DTLN → RNNoise → SpectralGate).  Specifying a name skips
        straight to that model (still falls back on error).
    sample_rate : int
        Pipeline sample rate passed to the model constructor.

    Returns
    -------
    EnhancementModel
        A ready-to-use model instance.
    """
    from ai.models.pretrained_dtln import DTLNModel, is_available as dtln_available
    from ai.models.pretrained_rnnoise import RNNoiseModel, is_available as rnnoise_available
    from ai.models.spectral_gate import SpectralGateModel

    # Build the ordered list of candidates
    if prefer == "dtln":
        candidates = [("dtln", dtln_available, DTLNModel)]
    elif prefer == "rnnoise":
        candidates = [("rnnoise", rnnoise_available, RNNoiseModel)]
    elif prefer == "spectral_gate":
        candidates = [("spectral_gate", lambda: True, SpectralGateModel)]
    else:  # auto
        candidates = [
            ("dtln", dtln_available, DTLNModel),
            ("rnnoise", rnnoise_available, RNNoiseModel),
            ("spectral_gate", lambda: True, SpectralGateModel),
        ]

    for name, check_fn, model_cls in candidates:
        if not check_fn():
            warnings.warn(f"[model-select] {name}: runtime not available, skipping.")
            continue
        try:
            if name == "dtln":
                model = model_cls(target_sample_rate=sample_rate)
                # Force lazy-load to verify checkpoint exists
                model._ensure_session()
            elif name == "rnnoise":
                model = model_cls(target_sample_rate=sample_rate)
                model._ensure_backend()
            else:
                model = model_cls(sample_rate=sample_rate)
            print(f"[model-select] Using model: {model.name}")
            return model
        except Exception as exc:
            warnings.warn(f"[model-select] {name}: failed to load — {exc}")
            continue

    # This should never happen because SpectralGate has no dependencies,
    # but just in case:
    print("[model-select] Falling back to SpectralGateModel (all others failed).")
    return SpectralGateModel(sample_rate=sample_rate)


__all__ = ["EnhancementModel", "get_best_available_model"]
