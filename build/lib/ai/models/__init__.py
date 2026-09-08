"""Pluggable speech enhancement model backends.

All models implement :class:`ai.models.base.EnhancementModel` so the
streaming engine and hardware integration code depend only on the interface,
never on a specific model implementation.

Use :func:`get_best_available_model` to get the best model that can actually
run on this machine — it walks the fallback chain:
    Fine-tuned DTLN → Stock DTLN → Conv-TasNet → RNNoise → SpectralGate
and returns the first one that loads successfully.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Literal

from ai.models.base import EnhancementModel


def _find_finetuned_checkpoint() -> Path | None:
    """Search for a fine-tuned DTLN checkpoint."""
    # Check environment variable
    env_path = os.environ.get("DTLN_FINETUNED_PATH", "")
    if env_path and Path(env_path).exists():
        return Path(env_path)

    # Check standard locations relative to repo root
    this_dir = Path(__file__).resolve().parent
    for parent in [this_dir, *this_dir.parents]:
        if (parent / "pyproject.toml").exists():
            candidates = [
                parent / "results" / "m8_01_train_enhancement_model" / "dtln" / "best_model.pt",
                parent / "models" / "dtln_finetuned" / "best_model.pt",
            ]
            for c in candidates:
                if c.exists():
                    return c
            break
    return None


def _find_conv_tasnet_checkpoint() -> Path | None:
    """Search for a trained Conv-TasNet checkpoint."""
    this_dir = Path(__file__).resolve().parent
    for parent in [this_dir, *this_dir.parents]:
        if (parent / "pyproject.toml").exists():
            candidates = [
                parent / "results" / "m8_01_train_enhancement_model" / "conv_tasnet" / "best_model.pt",
                parent / "models" / "conv_tasnet" / "best_model.pt",
            ]
            for c in candidates:
                if c.exists():
                    return c
            break
    return None


def get_best_available_model(
    *,
    prefer: Literal["auto", "dtln_finetuned", "dtln", "conv_tasnet", "rnnoise", "spectral_gate"] = "auto",
    sample_rate: int = 16_000,
) -> EnhancementModel:
    """Return the best available enhancement model.

    Parameters
    ----------
    prefer : str
        Which model to try first.  ``"auto"`` walks the full fallback
        chain (fine-tuned DTLN → stock DTLN → Conv-TasNet → RNNoise →
        SpectralGate).  Specifying a name skips straight to that model
        (still falls back on error).
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

    candidates: list[tuple[str, object, object]] = []

    if prefer == "auto":
        # Try fine-tuned DTLN first
        finetuned_path = _find_finetuned_checkpoint()
        if finetuned_path is not None:
            try:
                from ai.models.dtln_trainable import DTLNTrainableModel, is_available as torch_available
                if torch_available():
                    candidates.append(("dtln_finetuned", lambda: True,
                                       lambda: DTLNTrainableModel(checkpoint_path=finetuned_path, target_sample_rate=sample_rate)))
            except ImportError:
                pass

        candidates.extend([
            ("dtln", dtln_available, lambda: DTLNModel(target_sample_rate=sample_rate)),
        ])

        # Try Conv-TasNet if checkpoint exists
        tasnet_path = _find_conv_tasnet_checkpoint()
        if tasnet_path is not None:
            try:
                from ai.models.conv_tasnet import ConvTasNetModel
                candidates.append(("conv_tasnet", lambda: True,
                                    lambda: ConvTasNetModel(checkpoint_path=tasnet_path, target_sample_rate=sample_rate)))
            except ImportError:
                pass

        candidates.extend([
            ("rnnoise", rnnoise_available, lambda: RNNoiseModel(target_sample_rate=sample_rate)),
            ("spectral_gate", lambda: True, lambda: SpectralGateModel(sample_rate=sample_rate)),
        ])

    elif prefer == "dtln_finetuned":
        finetuned_path = _find_finetuned_checkpoint()
        if finetuned_path:
            from ai.models.dtln_trainable import DTLNTrainableModel
            candidates.append(("dtln_finetuned", lambda: True,
                                lambda: DTLNTrainableModel(checkpoint_path=finetuned_path, target_sample_rate=sample_rate)))
    elif prefer == "dtln":
        candidates = [("dtln", dtln_available, lambda: DTLNModel(target_sample_rate=sample_rate))]
    elif prefer == "conv_tasnet":
        tasnet_path = _find_conv_tasnet_checkpoint()
        if tasnet_path:
            from ai.models.conv_tasnet import ConvTasNetModel
            candidates.append(("conv_tasnet", lambda: True,
                                lambda: ConvTasNetModel(checkpoint_path=tasnet_path, target_sample_rate=sample_rate)))
    elif prefer == "rnnoise":
        candidates = [("rnnoise", rnnoise_available, lambda: RNNoiseModel(target_sample_rate=sample_rate))]
    elif prefer == "spectral_gate":
        candidates = [("spectral_gate", lambda: True, lambda: SpectralGateModel(sample_rate=sample_rate))]

    for name, check_fn, factory in candidates:
        if not check_fn():
            warnings.warn(f"[model-select] {name}: runtime not available, skipping.")
            continue
        try:
            model = factory()
            # Force lazy-load to verify the model actually works
            if hasattr(model, '_ensure_session'):
                model._ensure_session()
            elif hasattr(model, '_ensure_model'):
                model._ensure_model()
            elif hasattr(model, '_ensure_backend'):
                model._ensure_backend()
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

