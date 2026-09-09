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
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

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
        print("\n[WARN] No test windows found!")
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

    def _calc_stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {}
        arr = np.array(values, dtype=np.float64)
        return {
            "min": float(np.min(arr)),
            "q1": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "mean": float(np.mean(arr)),
            "q3": float(np.percentile(arr, 75)),
            "max": float(np.max(arr)),
            "std": float(np.std(arr)),
        }

    def _format_dist(stats: dict[str, float], unit: str = "") -> str:
        if not stats:
            return "N/A"
        return (
            f"mean={stats['mean']:+.2f}{unit}, med={stats['median']:+.2f}{unit}, "
            f"IQR=[{stats['q1']:+.2f}, {stats['q3']:+.2f}], "
            f"range=[{stats['min']:+.2f} .. {stats['max']:+.2f}]"
        )

    def _summarize(label: str, results: list) -> None:
        si_snrs = [r["si_snr_db"] for r in results if r.get("si_snr_db") is not None]
        stois = [r["stoi"] for r in results if r.get("stoi") is not None]
        pesqs = [r["pesq_approx"] for r in results if r.get("pesq_approx") is not None]

        print(f"\n--- {label} ({len(results)} windows) ---")
        if si_snrs:
            stats = _calc_stats(si_snrs)
            print(f"  SI-SNR (dB): {_format_dist(stats, ' dB')}")
        if stois:
            stats = _calc_stats(stois)
            print(f"  STOI:        {_format_dist(stats)}")
        if pesqs:
            stats = _calc_stats(pesqs)
            print(f"  PESQ:        {_format_dist(stats)}")

    _summarize("HELD-OUT (never seen in training)", held_out_results)
    _summarize("SEEN (in test but categories overlap with train)", seen_results)

    # Gap analysis
    ho_snrs = [r["si_snr_db"] for r in held_out_results if r.get("si_snr_db") is not None]
    seen_snrs = [r["si_snr_db"] for r in seen_results if r.get("si_snr_db") is not None]
    if ho_snrs and seen_snrs:
        gap = np.mean(seen_snrs) - np.mean(ho_snrs)
        print(f"\n{'='*70}")
        print(f"GENERALIZATION GAP: {gap:+.2f} dB SI-SNR")
        if abs(gap) < 2.0:
            print("[PASS] Gap < 2 dB -- strong generalization evidence.")
        elif abs(gap) < 5.0:
            print("[WARN] Gap 2-5 dB -- moderate, potentially acceptable.")
        else:
            print("[FAIL] Gap > 5 dB -- significant generalization concern.")

    # Phase C.2: Explicit Worst-Case Condition Reporting
    print(f"\n{'='*70}")
    print("WORST-CASE CONDITION ANALYSIS (Phase C.2 Requirement)")
    print(f"{'='*70}")
    all_test = held_out_results + seen_results
    if all_test:
        # Group by (noise_family, snr_db)
        condition_map: dict[tuple[str, float], list[dict]] = {}
        for r in all_test:
            nf = str(r.get("noise_family", "unknown"))
            snr = float(r.get("snr_db", 0.0))
            condition_map.setdefault((nf, snr), []).append(r)

        # Compute average SI-SNR per condition to identify hardest
        cond_scores = []
        for (nf, snr), cond_res in condition_map.items():
            vals = [x["si_snr_db"] for x in cond_res if x.get("si_snr_db") is not None]
            if vals:
                cond_scores.append(((nf, snr), np.mean(vals), cond_res))

        if cond_scores:
            cond_scores.sort(key=lambda x: x[1])  # lowest mean SI-SNR first
            (worst_nf, worst_snr), worst_mean, worst_res = cond_scores[0]
            worst_snrs = [x["si_snr_db"] for x in worst_res if x.get("si_snr_db") is not None]
            worst_stois = [x["stoi"] for x in worst_res if x.get("stoi") is not None]
            worst_pesqs = [x["pesq_approx"] for x in worst_res if x.get("pesq_approx") is not None]

            print(f"  Hardest Noise Family:  {worst_nf}")
            print(f"  Lowest Input SNR:      {worst_snr:+.1f} dB")
            print(f"  Worst-Case Condition:  family='{worst_nf}', SNR={worst_snr:+.1f} dB ({len(worst_res)} windows)")
            if worst_snrs:
                w_snr_stats = _calc_stats(worst_snrs)
                print(f"    Worst-Case SI-SNR:   {_format_dist(w_snr_stats, ' dB')}")
            if worst_stois:
                w_stoi_stats = _calc_stats(worst_stois)
                print(f"    Worst-Case STOI:     {_format_dist(w_stoi_stats)}")
            if worst_pesqs:
                w_pesq_stats = _calc_stats(worst_pesqs)
                print(f"    Worst-Case PESQ:     {_format_dist(w_pesq_stats)}")

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
