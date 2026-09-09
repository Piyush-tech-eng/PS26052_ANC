"""M10.01 — Robustness sweep under dynamic conditions.

Tests how each method's metrics degrade (or don't) under:
- Noise onset timing variations
- Overlapping noise events
- Mid-clip SNR transitions
- Stationary / nonstationary / impulsive / mixed taxonomy

This is the PS statement's explicit differentiator for AI over classical.

Usage::

    python experiments/m10_01_robustness_sweep.py \\
        --dataset results/m7_speech_ai_handoff \\
        --output results/m10_01_robustness_sweep
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _create_dynamic_scenario(
    clean: np.ndarray,
    noise: np.ndarray,
    scenario_type: str,
    sample_rate: int,
    seed: int = 42,
) -> np.ndarray:
    """Create a dynamic noise scenario by modifying the noise signal."""
    rng = np.random.default_rng(seed)
    n = len(clean)
    noise = noise[:n] if len(noise) >= n else np.pad(noise, (0, n - len(noise)))

    if scenario_type == "delayed_onset":
        # Noise starts at 30% of the signal
        onset = int(0.3 * n)
        noise[:onset] = 0.0

    elif scenario_type == "sudden_offset":
        # Noise stops at 70% of the signal
        offset = int(0.7 * n)
        noise[offset:] = 0.0

    elif scenario_type == "snr_transition":
        # SNR drops by 10dB at midpoint
        mid = n // 2
        noise[mid:] *= 3.16  # ~10dB increase

    elif scenario_type == "overlapping_events":
        # Add a second noise burst at random location
        burst_len = min(int(0.2 * n), n)
        burst_start = rng.integers(0, max(1, n - burst_len))
        burst = rng.normal(0, 0.5, burst_len)
        noise[burst_start:burst_start + burst_len] += burst

    elif scenario_type == "intermittent":
        # Noise alternates on/off every 0.5 seconds
        period = int(0.5 * sample_rate)
        for i in range(0, n, 2 * period):
            end = min(i + period, n)
            noise[i:end] = 0.0

    return clean + noise


def run_robustness_sweep(
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Run robustness sweep across dynamic conditions."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    from anc.evaluation.dataset import load_dataset

    splits, _ = load_dataset(dataset_dir)
    test_windows = splits.get("test", [])

    if not test_windows:
        print("ERROR: No test windows found.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    scenario_types = [
        "static",          # Baseline: no dynamic changes
        "delayed_onset",   # Noise starts partway through
        "sudden_offset",   # Noise stops partway through
        "snr_transition",  # SNR change mid-clip
        "overlapping_events",  # Multiple noise sources
        "intermittent",    # On/off noise pattern
    ]

    rows: list[dict[str, object]] = []

    # Get best available model
    try:
        from ai.models import get_best_available_model
        model = get_best_available_model()
        model_name = model.name
    except Exception:
        model = None
        model_name = "unavailable"

    for window in test_windows[:50]:  # Limit for tractability
        clean = window.clean_speech_target if window.clean_speech_target is not None else window.reference_x
        noise_comp = window.noise_component if window.noise_component is not None else np.zeros_like(clean)

        for scenario_type in scenario_types:
            if scenario_type == "static":
                test_signal = window.noisy_speech if window.noisy_speech is not None else window.baseline_residual
            else:
                test_signal = _create_dynamic_scenario(
                    clean.copy(), noise_comp.copy(), scenario_type, window.sample_rate
                )

            # Evaluate with AI model
            if model is not None:
                try:
                    enhanced = model.enhance(test_signal)
                except Exception:
                    enhanced = test_signal.copy()
            else:
                enhanced = test_signal.copy()

            # Compute metrics
            from anc.evaluation.metrics import compute_noise_reduction_db
            snr = compute_noise_reduction_db(test_signal, enhanced - clean)

            stoi_val = 0.0
            try:
                from pystoi import stoi
                stoi_val = float(stoi(clean, enhanced, window.sample_rate, extended=False))
            except Exception:
                pass

            noise_family = "unknown"
            if window.speech_metadata:
                noise_family = str(window.speech_metadata.get("noise_family", "unknown"))

            rows.append({
                "scenario_type": scenario_type,
                "noise_family": noise_family,
                "model": model_name,
                "snr_db": snr,
                "stoi": stoi_val,
                "window_id": window.window_id,
            })

    # Write results
    csv_path = output_path / "robustness_sweep.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Summary: average metrics per scenario type
    summary: dict[str, dict[str, float]] = {}
    for row in rows:
        st = str(row["scenario_type"])
        summary.setdefault(st, {"snr_values": [], "stoi_values": []})
        summary[st]["snr_values"].append(row["snr_db"])
        summary[st]["stoi_values"].append(row["stoi"])

    summary_report = {}
    for st, vals in summary.items():
        summary_report[st] = {
            "mean_snr": float(np.mean(vals["snr_values"])) if vals["snr_values"] else 0,
            "mean_stoi": float(np.mean(vals["stoi_values"])) if vals["stoi_values"] else 0,
            "count": len(vals["snr_values"]),
        }

    summary_path = output_path / "robustness_summary.json"
    summary_path.write_text(json.dumps(summary_report, indent=2) + "\n", encoding="utf-8")

    print(f"Robustness sweep: {len(rows)} evaluations -> {csv_path}")
    print("\nSummary by scenario type:")
    for st, stats in summary_report.items():
        print(f"  {st:>25s}: SNR={stats['mean_snr']:.2f}dB, STOI={stats['mean_stoi']:.4f} (n={stats['count']})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness sweep.")
    parser.add_argument("--dataset", default="results/m7_speech_ai_handoff")
    parser.add_argument("--output", default="results/m10_01_robustness_sweep")
    args = parser.parse_args()

    run_robustness_sweep(args.dataset, args.output)


if __name__ == "__main__":
    main()
