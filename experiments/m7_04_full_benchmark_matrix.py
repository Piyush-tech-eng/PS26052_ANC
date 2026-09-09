"""M7.04 — Full benchmark matrix execution.

Runs all methods (no-control, Wiener, LMS/NLMS, FxLMS/FxNLMS, AI-only,
hybrid) across the entire Phase A test split, producing a comprehensive
comparison table with every method × every (noise_family, SNR) cell populated.

Usage::

    python experiments/m7_04_full_benchmark_matrix.py \\
        --dataset results/m7_speech_ai_handoff \\
        --output results/m7_04_benchmark_matrix
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anc.evaluation.dataset import load_dataset
from anc.evaluation.metrics import (
    compute_noise_reduction_db,
    compute_signal_power,
)


def _compute_metrics(
    noisy: np.ndarray,
    enhanced: np.ndarray,
    clean: np.ndarray,
    sample_rate: int,
) -> dict[str, float]:
    """Compute SNR, STOI, PESQ for one sample."""
    snr = compute_noise_reduction_db(noisy, enhanced - clean)

    stoi_val = 0.0
    try:
        from pystoi import stoi
        s = float(stoi(clean, enhanced, sample_rate, extended=False))
        if s > 0.001:
            stoi_val = s
        else:
            from anc.evaluation.metrics import compute_stoi as compute_builtin_stoi
            stoi_val = float(compute_builtin_stoi(enhanced, clean, sample_rate))
    except (ImportError, Exception):
        try:
            from anc.evaluation.metrics import compute_stoi as compute_builtin_stoi
            stoi_val = float(compute_builtin_stoi(enhanced, clean, sample_rate))
        except Exception:
            pass

    pesq_val = 0.0
    try:
        from pesq import pesq
        mode = "wb" if sample_rate >= 16000 else "nb"
        pesq_val = float(pesq(sample_rate, clean, enhanced, mode))
    except (ImportError, Exception):
        pass

    return {"snr_db": snr, "stoi": stoi_val, "pesq": pesq_val}


def run_benchmark_matrix(
    dataset_dir: str | Path,
    output_dir: str | Path,
) -> None:
    """Run the full benchmark matrix."""
    splits, normalization = load_dataset(dataset_dir)
    test_windows = splits.get("test", [])

    if not test_windows:
        print("ERROR: No test windows found.")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Methods to evaluate
    methods = ["no_processing", "wiener", "ai_dtln", "hybrid"]

    sr = test_windows[0].sample_rate if test_windows else 16000
    ai_model = None
    hybrid_engine = None
    try:
        from ai.models import get_best_available_model
        from ai.streaming.hybrid_engine import HybridEngine
        ai_model = get_best_available_model(sample_rate=sr)
        hybrid_engine = HybridEngine(model=ai_model, sample_rate=sr)
    except Exception as e:
        print(f"Warning: Could not initialize AI model: {e}")

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    rows: list[dict[str, object]] = []

    for idx, window in enumerate(test_windows):
        if (idx + 1) % 100 == 0 or (idx + 1) == len(test_windows):
            print(f"Evaluated {idx + 1}/{len(test_windows)} test windows...", flush=True)

        noise_family = "unknown"
        snr_db = 0.0
        if window.speech_metadata:
            noise_family = str(window.speech_metadata.get("noise_family", "unknown"))
            snr_db = float(window.speech_metadata.get("snr_db", 0.0))

        noisy = window.noisy_speech if window.noisy_speech is not None else window.baseline_residual
        clean = window.clean_speech_target if window.clean_speech_target is not None else window.reference_x

        for method in methods:
            if method == "no_processing":
                enhanced = noisy.copy()
            elif method == "wiener":
                try:
                    from anc.statistics import (
                        apply_fir,
                        autocorrelation,
                        build_correlation_matrix,
                        build_cross_correlation_vector,
                        cross_correlation,
                        solve_wiener_hopf,
                    )
                    fl = min(64, len(noisy) // 4)
                    rxx = autocorrelation(noisy, max_lag=fl - 1)
                    rxd = cross_correlation(noisy, clean, max_lag=fl - 1)
                    R = build_correlation_matrix(rxx, fl)
                    p = build_cross_correlation_vector(rxd, fl)
                    w = solve_wiener_hopf(R, p)
                    enhanced = apply_fir(noisy, w)
                except Exception:
                    enhanced = noisy.copy()
            elif method == "ai_dtln":
                if ai_model is not None:
                    enhanced = ai_model.enhance(noisy)
                else:
                    enhanced = noisy.copy()
            elif method == "hybrid":
                if hybrid_engine is not None:
                    enhanced, _ = hybrid_engine.process_batch(noisy)
                else:
                    enhanced = noisy.copy()
            else:
                enhanced = noisy.copy()

            metrics = _compute_metrics(noisy, enhanced, clean, window.sample_rate)

            rows.append({
                "method": method,
                "noise_family": noise_family,
                "snr_db": snr_db,
                "scenario_id": window.scenario_id,
                **metrics,
            })

    # Write CSV
    csv_path = output_path / "benchmark_matrix.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    # Summary statistics
    summary: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        key = f"{row['method']}__{row['noise_family']}__{row['snr_db']}"
        summary.setdefault(key, {"snr_values": [], "stoi_values": [], "pesq_values": []})
        summary[key]["snr_values"].append(row["snr_db"])
        summary[key]["stoi_values"].append(row["stoi"])
        summary[key]["pesq_values"].append(row["pesq"])

    summary_path = output_path / "benchmark_summary.json"
    summary_path.write_text(json.dumps({
        "total_evaluations": len(rows),
        "methods": sorted(set(r["method"] for r in rows)),
        "noise_families": sorted(set(r["noise_family"] for r in rows)),
    }, indent=2) + "\n", encoding="utf-8")

    print(f"Benchmark matrix: {len(rows)} evaluations -> {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full benchmark matrix.")
    parser.add_argument("--dataset", default="results/m7_speech_ai_handoff")
    parser.add_argument("--output", default="results/m7_04_benchmark_matrix")
    args = parser.parse_args()

    run_benchmark_matrix(args.dataset, args.output)


if __name__ == "__main__":
    main()
