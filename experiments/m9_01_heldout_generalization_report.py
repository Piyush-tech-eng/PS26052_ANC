"""M9.01 — Held-out generalization report.

Runs Phase 4's batch metrics specifically on speakers and noise families
held entirely out of train/validation.  This produces the table that proves
generalization rather than memorization.

Usage::

    python experiments/m9_01_heldout_generalization_report.py \\
        --dataset results/m7_speech_ai_handoff \\
        --output results/heldout_generalization_report.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anc.evaluation.dataset import load_dataset
from m7_03_batch_metrics_report import compute_window_metrics


def run_generalization_report(
    dataset_dir: str | Path,
    output_path: str | Path | None = None,
) -> None:
    """Evaluate held-out test data and print generalization evidence.

    Analyzes speakers and noise families in the test split that are
    absent from train+validation, proving the model generalizes.
    """
    splits, _ = load_dataset(dataset_dir)

    # Identify speakers/noise families in each split
    split_speakers: dict[str, set[str]] = {}
    split_noises: dict[str, set[str]] = {}

    for split_name, windows in splits.items():
        speakers = set()
        noises = set()
        for w in windows:
            if w.speech_metadata:
                spk = w.speech_metadata.get("speech_source_id")
                nf = w.speech_metadata.get("noise_family")
                if spk:
                    speakers.add(str(spk))
                if nf:
                    noises.add(str(nf))
        split_speakers[split_name] = speakers
        split_noises[split_name] = noises

    train_val_speakers = split_speakers.get("train", set()) | split_speakers.get("validation", set())
    train_val_noises = split_noises.get("train", set()) | split_noises.get("validation", set())
    test_speakers = split_speakers.get("test", set())
    test_noises = split_noises.get("test", set())

    held_out_speakers = test_speakers - train_val_speakers
    held_out_noises = test_noises - train_val_noises

    print("=" * 70)
    print("HELD-OUT GENERALIZATION REPORT")
    print("=" * 70)
    print(f"\nSpeakers in train+val: {sorted(train_val_speakers)}")
    print(f"Speakers in test:     {sorted(test_speakers)}")
    print(f"HELD-OUT speakers:    {sorted(held_out_speakers)}")
    print(f"\nNoise families in train+val: {sorted(train_val_noises)}")
    print(f"Noise families in test:     {sorted(test_noises)}")
    print(f"HELD-OUT noise families:    {sorted(held_out_noises)}")

    # Run metrics on test split
    test_windows = splits.get("test", [])
    if not test_windows:
        print("\n⚠ No test windows found!")
        return

    sr = test_windows[0].sample_rate

    # Separate held-out vs seen
    held_out_results = []
    seen_results = []

    for w in test_windows:
        metrics = compute_window_metrics(w, sampling_rate=sr)
        spk = metrics.get("speech_source_id")
        nf = metrics.get("noise_family")

        is_held_out = (
            (spk and str(spk) in held_out_speakers) or
            (nf and str(nf) in held_out_noises)
        )

        if is_held_out:
            held_out_results.append(metrics)
        else:
            seen_results.append(metrics)

    def _summarize(label: str, results: list) -> None:
        si_snrs = [r["si_snr_db"] for r in results if r["si_snr_db"] is not None]
        stois = [r["stoi"] for r in results if r["stoi"] is not None]
        pesqs = [r["pesq_approx"] for r in results if r["pesq_approx"] is not None]

        print(f"\n--- {label} ({len(results)} windows) ---")
        if si_snrs:
            print(f"  SI-SNR:  {np.mean(si_snrs):+.2f} ± {np.std(si_snrs):.2f} dB")
        if stois:
            print(f"  STOI:    {np.mean(stois):.3f} ± {np.std(stois):.3f}")
        if pesqs:
            print(f"  PESQ:    {np.mean(pesqs):.2f} ± {np.std(pesqs):.2f}")

    _summarize("HELD-OUT (never seen in training)", held_out_results)
    _summarize("SEEN (in test but categories overlap with train)", seen_results)

    # Gap analysis
    ho_snrs = [r["si_snr_db"] for r in held_out_results if r["si_snr_db"] is not None]
    seen_snrs = [r["si_snr_db"] for r in seen_results if r["si_snr_db"] is not None]
    if ho_snrs and seen_snrs:
        gap = np.mean(seen_snrs) - np.mean(ho_snrs)
        print(f"\n{'='*70}")
        print(f"GENERALIZATION GAP: {gap:+.2f} dB SI-SNR")
        if abs(gap) < 2.0:
            print("✅ Gap < 2 dB — strong generalization evidence.")
        elif abs(gap) < 5.0:
            print("⚠️  Gap 2-5 dB — moderate, potentially acceptable.")
        else:
            print("❌ Gap > 5 dB — significant generalization concern.")

    # Write detailed report
    if output_path:
        import csv
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        all_results = held_out_results + seen_results
        fieldnames = [
            "window_id", "scenario_id", "source_type", "controller",
            "speech_source_id", "noise_family", "snr_db",
            "si_snr_db", "stoi", "pesq_approx", "held_out",
        ]
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in held_out_results:
                row = {k: r.get(k) for k in fieldnames if k != "held_out"}
                row["held_out"] = True
                writer.writerow(row)
            for r in seen_results:
                row = {k: r.get(k) for k in fieldnames if k != "held_out"}
                row["held_out"] = False
                writer.writerow(row)

        print(f"\nDetailed report written to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Held-out generalization report")
    parser.add_argument("--dataset", required=True, help="Dataset directory")
    parser.add_argument("--output", default=None, help="Output CSV path")
    args = parser.parse_args()

    run_generalization_report(args.dataset, args.output)
