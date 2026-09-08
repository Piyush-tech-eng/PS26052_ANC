"""PyTorch Dataset/DataLoader wrapping the PS26052 ANC evaluation dataset.

Bridges the existing ``anc.evaluation.load_dataset()`` / ``DatasetWindow``
infrastructure to PyTorch's training data pipeline, applying:

1. Frozen z-score normalization (from ``apply_normalization()``, using
   training-set statistics only — no leakage).
2. Training-time random augmentation (gain jitter, time shift — separate
   from the dataset-level augmentation in Phase A.4).

This keeps the ANC evaluation code framework-agnostic while giving the
PyTorch trainer exactly the tensors it needs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from anc.evaluation.dataset import DatasetWindow, load_dataset, apply_normalization


class PS26052Dataset:
    """PyTorch Dataset wrapping PS26052 ANC DatasetWindows.

    Parameters
    ----------
    windows : list of DatasetWindow
        The dataset windows to wrap.
    normalization : dict
        Normalization statistics from ``compute_normalization()``.
    apply_norm : bool
        Whether to apply z-score normalization.
    augment : bool
        Whether to apply training-time random augmentation.
    seed : int
        Random seed for augmentation reproducibility.
    """

    def __init__(
        self,
        windows: list[DatasetWindow],
        normalization: dict[str, Any],
        *,
        apply_norm: bool = True,
        augment: bool = False,
        seed: int = 42,
    ) -> None:
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PS26052Dataset.")

        self._windows = windows
        self._normalization = normalization
        self._apply_norm = apply_norm
        self._augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        window = self._windows[idx]

        # Apply normalization if requested
        if self._apply_norm:
            window = apply_normalization(window, self._normalization)

        # Get input and target arrays
        if window.noisy_speech is not None:
            noisy = window.noisy_speech.copy()
        else:
            noisy = window.baseline_residual.copy()

        if window.clean_speech_target is not None:
            clean = window.clean_speech_target.copy()
        elif window.target is not None:
            clean = window.target.copy()
        else:
            # Fallback: use reference as target (identity mapping)
            clean = window.reference_x.copy()

        # Training-time augmentation
        if self._augment:
            noisy, clean = self._apply_augmentation(noisy, clean)

        return {
            "noisy": torch.from_numpy(noisy).float(),
            "clean": torch.from_numpy(clean).float(),
            "scenario_id": window.scenario_id,
            "window_id": window.window_id,
        }

    def _apply_augmentation(
        self,
        noisy: np.ndarray,
        clean: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply training-time random augmentation.

        Separate from dataset-level augmentation (Phase A.4).
        Applied identically to both noisy and clean to maintain alignment.
        """
        # Random gain jitter (±6 dB)
        gain_db = self._rng.uniform(-6.0, 6.0)
        gain_linear = 10.0 ** (gain_db / 20.0)
        noisy = noisy * gain_linear
        clean = clean * gain_linear

        # Random time shift (up to 10% of signal length)
        max_shift = max(1, len(noisy) // 10)
        shift = self._rng.integers(-max_shift, max_shift + 1)
        if shift != 0:
            noisy = np.roll(noisy, shift)
            clean = np.roll(clean, shift)

        return noisy, clean


def create_dataloaders(
    dataset_dir: str | Path,
    *,
    batch_size: int = 16,
    num_workers: int = 0,
    augment_train: bool = True,
    seed: int = 42,
) -> dict[str, Any]:
    """Create train/validation/test DataLoaders from a saved dataset.

    Parameters
    ----------
    dataset_dir : str or Path
        Path to the saved dataset directory (from ``save_dataset()``).
    batch_size : int
        Batch size for training.
    num_workers : int
        Number of data loading workers.
    augment_train : bool
        Whether to augment training data.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: ``"train"``, ``"validation"``, ``"test"`` → DataLoader instances.
        Also includes ``"normalization"`` for reference.
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for create_dataloaders.")

    splits, normalization = load_dataset(dataset_dir)

    loaders = {}
    for split_name, windows in splits.items():
        if not windows:
            continue

        dataset = PS26052Dataset(
            windows,
            normalization,
            apply_norm=True,
            augment=(split_name == "train" and augment_train),
            seed=seed,
        )

        loader = DataLoader(
            dataset,
            batch_size=batch_size if split_name == "train" else batch_size * 2,
            shuffle=(split_name == "train"),
            num_workers=num_workers,
            drop_last=(split_name == "train"),
        )

        loaders[split_name] = loader

    loaders["normalization"] = normalization
    return loaders
