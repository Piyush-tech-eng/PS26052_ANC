"""Hyperparameter search for PS26052 speech enhancement training.

Grid or random search over learning rate, loss-term weights, and batch size.
Each run's config and resulting metrics are logged for the ablation table.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ai.training.trainer import TrainingConfig, train_model


@dataclass
class SearchSpace:
    """Defines the hyperparameter search space.

    Each field is a list of values to try.  The search produces the
    Cartesian product of all fields (grid search) or a random subset.
    """

    learning_rates: list[float] = None
    si_snr_weights: list[float] = None
    l1_weights: list[float] = None
    stft_weights: list[float] = None
    batch_sizes: list[int] = None

    def __post_init__(self) -> None:
        if self.learning_rates is None:
            self.learning_rates = [1e-4, 5e-4, 1e-3]
        if self.si_snr_weights is None:
            self.si_snr_weights = [1.0]
        if self.l1_weights is None:
            self.l1_weights = [0.0, 0.1, 0.5]
        if self.stft_weights is None:
            self.stft_weights = [0.0, 0.5, 1.0]
        if self.batch_sizes is None:
            self.batch_sizes = [8, 16]


from dataclasses import dataclass, field


def grid_search(
    model_factory: Any,
    train_loader_factory: Any,
    val_loader: Any,
    search_space: SearchSpace,
    output_dir: str | Path,
    *,
    max_epochs_per_run: int = 50,
    sample_rate: int = 16_000,
    device: str = "cpu",
) -> list[dict[str, Any]]:
    """Run grid search over hyperparameter combinations.

    Parameters
    ----------
    model_factory : callable
        Function that returns a fresh model instance.
    train_loader_factory : callable(batch_size) -> DataLoader
        Function that returns a DataLoader for the given batch size.
    val_loader : DataLoader
        Validation DataLoader.
    search_space : SearchSpace
        Hyperparameter search space.
    output_dir : str or Path
        Root output directory for all runs.
    max_epochs_per_run : int
        Maximum epochs per training run.
    sample_rate : int
        Audio sample rate.
    device : str
        Training device.

    Returns
    -------
    list of dict
        Results for each hyperparameter combination, sorted by best STOI.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    combinations = list(itertools.product(
        search_space.learning_rates,
        search_space.si_snr_weights,
        search_space.l1_weights,
        search_space.stft_weights,
        search_space.batch_sizes,
    ))

    print(f"\nHyperparameter search: {len(combinations)} combinations")
    results: list[dict[str, Any]] = []

    for run_idx, (lr, si_snr_w, l1_w, stft_w, batch_size) in enumerate(combinations):
        run_name = f"run_{run_idx:03d}_lr{lr:.0e}_si{si_snr_w}_l1{l1_w}_stft{stft_w}_bs{batch_size}"
        run_dir = output_path / run_name

        print(f"\n{'─'*60}")
        print(f"Run {run_idx + 1}/{len(combinations)}: {run_name}")
        print(f"{'─'*60}")

        config = TrainingConfig(
            learning_rate=lr,
            si_snr_weight=si_snr_w,
            l1_weight=l1_w,
            stft_weight=stft_w,
            max_epochs=max_epochs_per_run,
            patience=10,  # Shorter patience for search
        )

        model = model_factory()
        train_loader = train_loader_factory(batch_size)

        try:
            run_result = train_model(
                model, train_loader, val_loader, config, run_dir,
                sample_rate=sample_rate, device=device,
            )

            result = {
                "run_name": run_name,
                "config": {
                    "learning_rate": lr,
                    "si_snr_weight": si_snr_w,
                    "l1_weight": l1_w,
                    "stft_weight": stft_w,
                    "batch_size": batch_size,
                },
                "best_stoi": run_result["best_stoi"],
                "best_epoch": run_result["best_epoch"],
                "total_epochs": run_result["total_epochs"],
                "final_metrics": run_result["final_metrics"],
                "checkpoint_path": run_result["checkpoint_path"],
            }
        except Exception as exc:
            result = {
                "run_name": run_name,
                "config": {
                    "learning_rate": lr,
                    "si_snr_weight": si_snr_w,
                    "l1_weight": l1_w,
                    "stft_weight": stft_w,
                    "batch_size": batch_size,
                },
                "error": str(exc),
                "best_stoi": -float("inf"),
            }
            print(f"  ✗ Run failed: {exc}")

        results.append(result)

    # Sort by best STOI (descending)
    results.sort(key=lambda r: r.get("best_stoi", -float("inf")), reverse=True)

    # Save search results
    summary_path = output_path / "search_results.json"
    summary_path.write_text(
        json.dumps(results, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print("Hyperparameter search complete.")
    print(f"Best run: {results[0]['run_name']} (STOI={results[0].get('best_stoi', 'N/A')})")
    print(f"Results saved to: {summary_path}")
    print(f"{'='*60}")

    return results
