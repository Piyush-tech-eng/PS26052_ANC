"""M7.03 — Batch metrics report over an entire dataset split.

Runs SI-SNR / STOI / PESQ-approx over every window in a specified split
and writes a CSV/table.  This is the backbone artifact for the judge-facing
metrics table and Phase 9's generalization proof.

Usage::

    python experiments/m7_03_batch_metrics_report.py \\
        --dataset results/m7_speech_ai_handoff \\
        --split test \\
        --output results/batch_metrics_report.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anc.evaluation.dataset import load_dataset
from anc.evaluation.metrics import (
    compute_si_snr,
    compute_stoi,
    compute_pesq_approx,
    compute_signal_power,
    compute_noise_reduction_db,
)


def compute_window_metrics(
    window,
    sampling_rate: int = 8000,
) -> dict[str, object]:
    """Compute metrics for a single dataset window."""
    metrics: dict[str, object] = {
        "window_id": window.window_id,
        "scenario_id": window.scenario_id,
        "source_type": window.source_type,
        "controller": window.algorithm_metadata.get("controller", "unknown"),
    }

    # Speech metadata if available
    if window.speech_metadata:
        metrics["speech_source_id"] = window.speech_metadata.get("speech_source_id")
        metrics["noise_family"] = window.speech_metadata.get("noise_family")
        metrics["snr_db"] = window.speech_metadata.get("snr_db")
    else:
        metrics["speech_source_id"] = None
        metrics["noise_family"] = None
        metrics["snr_db"] = None

    # Power metrics
    metrics["residual_power"] = compute_signal_power(window.baseline_residual)

    # Target-based metrics
    if window.target is not None:
        target = window.target
        residual = window.baseline_residual
        metrics["si_snr_db"] = compute_si_snr(residual, target)
        metrics["stoi"] = compute_stoi(residual, target, sampling_rate)
        metrics["pesq_approx"] = compute_pesq_approx(residual, target, sampling_rate)
    elif window.clean_speech_target is not None:
        clean = window.clean_speech_target
        noisy = window.noisy_speech if window.noisy_speech is not None else window.baseline_residual
        metrics["si_snr_db"] = compute_si_snr(noisy, clean)
        metrics["stoi"] = compute_stoi(noisy, clean, sampling_rate)
        metrics["pesq_approx"] = compute_pesq_approx(noisy, clean, sampling_rate)
    else:
        metrics["si_snr_db"] = None
        metrics["stoi"] = None
        metrics["pesq_approx"] = None

    return metrics


def run_batch_report(
    dataset_dir: str | Path,
    split: str = "test",
    output_path: str | Path | None = None,
) -> list[dict[str, object]]:
    """Run metrics over all windows in a dataset split.

    Parameters
    ----------
    dataset_dir : path
        Path to the dataset directory (containing ``dataset_manifest.json``).
    split : str
        Which split to evaluate (``"train"``, ``"validation"``, or ``"test"``).
    output_path : path, optional
        If provided, write results to this CSV file.

    Returns
    -------
    list of dict
        Per-window metrics.
    """
    splits, normalization = load_dataset(dataset_dir)

    if split not in splits:
        available = ", ".join(splits.keys())
        raise ValueError(f"Split '{split}' not found. Available: {available}")

    windows = splits[split]
    print(f"Evaluating {len(windows)} windows from '{split}' split...")

    # Determine sample rate from first window
    sr = windows[0].sample_rate if windows else 8000

    results = []
    for i, window in enumerate(windows):
        metrics = compute_window_metrics(window, sampling_rate=sr)
        results.append(metrics)

        if (i + 1) % 50 == 0 or i == len(windows) - 1:
            print(f"  Processed {i + 1}/{len(windows)} windows")

    # Print summary
    si_snrs = [r["si_snr_db"] for r in results if r["si_snr_db"] is not None]
    stois = [r["stoi"] for r in results if r["stoi"] is not None]
    pesqs = [r["pesq_approx"] for r in results if r["pesq_approx"] is not None]

    print(f"\n{'='*60}")
    print(f"BATCH METRICS REPORT — {split.upper()} split ({len(windows)} windows)")
    print(f"{'='*60}")

    if si_snrs:
        print(f"  SI-SNR:  mean={np.mean(si_snrs):+.2f} dB, "
              f"std={np.std(si_snrs):.2f}, min={np.min(si_snrs):+.2f}, "
              f"max={np.max(si_snrs):+.2f}")
    if stois:
        print(f"  STOI:    mean={np.mean(stois):.3f}, "
              f"std={np.std(stois):.3f}, min={np.min(stois):.3f}, "
              f"max={np.max(stois):.3f}")
    if pesqs:
        print(f"  PESQ:    mean={np.mean(pesqs):.2f}, "
              f"std={np.std(pesqs):.2f}, min={np.min(pesqs):.2f}, "
              f"max={np.max(pesqs):.2f}")

    # Per-noise-family breakdown
    families = set(r["noise_family"] for r in results if r["noise_family"] is not None)
    if families:
        print(f"\nPer noise family:")
        for fam in sorted(families):
            fam_snrs = [r["si_snr_db"] for r in results
                       if r["noise_family"] == fam and r["si_snr_db"] is not None]
            if fam_snrs:
                print(f"  {fam:>15s}: SI-SNR mean={np.mean(fam_snrs):+.2f} dB "
                      f"(n={len(fam_snrs)})")

    # Write CSV
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = [
            "window_id", "scenario_id", "source_type", "controller",
            "speech_source_id", "noise_family", "snr_db",
            "residual_power", "si_snr_db", "stoi", "pesq_approx",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow({k: row.get(k) for k in fieldnames})

        print(f"\nResults written to {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch metrics report")
    parser.add_argument("--dataset", required=True, help="Dataset directory")
    parser.add_argument("--split", default="test", help="Split to evaluate")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    run_batch_report(args.dataset, args.split, args.output)
