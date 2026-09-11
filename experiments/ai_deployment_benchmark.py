#!/usr/bin/env python3
"""PS26052 ANC — AI Precision & Deployment Benchmark.

Compares FP32, FP16, and INT8 quantized ONNX models:
- Model disk size (MB)
- Per-frame inference latency (mean, p95, p99 in ms)
- Real-time processing ratio
- Quality metrics (SNR improvement, noise reduction, output correlation)

Saves results to results/ai_deployment/ai_deployment_comparison.json.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

from ai.models.pretrained_dtln import DTLNModel
from ai.export.quantization import quantize_onnx


def get_model_size_mb(model_dir: Path) -> float:
    """Return total size in MB of all ONNX files in directory."""
    if not model_dir.exists():
        return 0.0
    total_bytes = sum(f.stat().st_size for f in model_dir.glob("*.onnx"))
    return round(total_bytes / (1024 * 1024), 2)


def generate_speech_noise_signal(sample_rate: int = 16_000, duration_s: float = 3.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic clean speech (harmonic formants) + ambient noise."""
    n = int(sample_rate * duration_s)
    t = np.arange(n) / sample_rate

    # Synthetic vowel-like harmonics
    clean = (
        0.5 * np.sin(2 * np.pi * 220 * t)
        + 0.3 * np.sin(2 * np.pi * 440 * t)
        + 0.2 * np.sin(2 * np.pi * 880 * t)
        + 0.1 * np.sin(2 * np.pi * 1320 * t)
    )
    # Apply envelope (syllable modulation)
    envelope = 0.5 * (1.0 + np.sin(2 * np.pi * 3 * t))
    clean = clean * envelope

    # Synthetic factory / HVAC noise (low-frequency + broadband)
    noise = 0.4 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.random.randn(n)
    noisy = clean + noise

    return clean.astype(np.float64), noise.astype(np.float64), noisy.astype(np.float64)


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio in dB."""
    s_pwr = float(np.mean(signal ** 2))
    n_pwr = float(np.mean(noise ** 2))
    if n_pwr <= 1e-12:
        return 60.0
    return float(10.0 * np.log10(max(s_pwr, 1e-12) / n_pwr))


def benchmark_tier(
    tier_name: str,
    model_dir: Path,
    clean: np.ndarray,
    noisy: np.ndarray,
    sample_rate: int = 16_000,
    warmup_runs: int = 2,
    num_runs: int = 5,
) -> dict[str, Any]:
    """Profile latency and enhancement quality for a single precision tier."""
    if not model_dir.exists():
        return {"error": f"Model directory not found: {model_dir}"}

    model = DTLNModel(model_path=model_dir, target_sample_rate=sample_rate)
    size_mb = get_model_size_mb(model_dir)

    # Warmup
    for _ in range(warmup_runs):
        _ = model.enhance(noisy[:sample_rate])

    # Latency profiling over full clip
    dur_s = len(noisy) / sample_rate
    latencies = []
    enhanced = None

    for _ in range(num_runs):
        t0 = time.perf_counter()
        enhanced = model.enhance(noisy)
        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)

    mean_elapsed = float(np.mean(latencies))
    rt_ratio = mean_elapsed / dur_s

    # Per-frame (20ms / 320 samples) latency equivalent
    num_frames = len(noisy) / 320
    per_frame_ms = (mean_elapsed / num_frames) * 1000.0

    # Quality metrics
    # Noise residual
    residual_noise = enhanced - clean
    in_noise = noisy - clean
    in_snr = compute_snr(clean, in_noise)
    out_snr = compute_snr(clean, residual_noise)
    snr_gain = out_snr - in_snr

    # Correlation with clean speech
    corr_matrix = np.corrcoef(clean, enhanced)
    speech_correlation = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0

    return {
        "tier": tier_name,
        "model_dir": str(model_dir),
        "size_mb": size_mb,
        "clip_duration_s": dur_s,
        "total_latency_s": round(mean_elapsed, 4),
        "per_frame_latency_ms": round(per_frame_ms, 2),
        "realtime_ratio": round(rt_ratio, 3),
        "realtime_multiplier": f"{round(1.0 / max(rt_ratio, 1e-4), 1)}x faster than real-time",
        "input_snr_db": round(in_snr, 2),
        "output_snr_db": round(out_snr, 2),
        "snr_improvement_db": round(snr_gain, 2),
        "speech_correlation": round(speech_correlation, 4),
        "deployable_edge": bool(rt_ratio < 0.7),
    }


def run_deployment_benchmark(output_path: str | Path = "results/ai_deployment/ai_deployment_comparison.json") -> dict[str, Any]:
    """Execute comparative benchmark across FP32, FP16, and INT8 tiers."""
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    fp32_dir = repo_root / "models" / "dtln"
    fp16_dir = repo_root / "models" / "dtln_fp16"
    int8_dir = repo_root / "models" / "dtln_quantized"

    # Ensure FP16 directory exists
    if not (fp16_dir / "model_1.onnx").exists():
        print("Converting models to FP16...")
        quantize_onnx(fp32_dir / "model_1.onnx", fp16_dir / "model_1.onnx", quantization_type="fp16")
        quantize_onnx(fp32_dir / "model_2.onnx", fp16_dir / "model_2.onnx", quantization_type="fp16")

    sample_rate = 16_000
    clean, noise, noisy = generate_speech_noise_signal(sample_rate=sample_rate, duration_s=3.0)

    print("\n================================================================================")
    print("PS26052 ANC - AI PRECISION & DEPLOYMENT BENCHMARK (FP32 vs FP16 vs INT8)")
    print("================================================================================")

    tiers = [
        ("FP32 (Baseline)", fp32_dir),
        ("FP16 (Half-Precision)", fp16_dir),
        ("INT8 (Quantized)", int8_dir),
    ]

    results = []
    for name, m_dir in tiers:
        print(f"Profiling {name}...")
        res = benchmark_tier(name, m_dir, clean, noisy, sample_rate=sample_rate)
        results.append(res)

    print("\n" + "=" * 80)
    print(f"{'Precision Tier':<22} | {'Size (MB)':<9} | {'Frame (ms)':<10} | {'RT Ratio':<9} | {'SNR Gain (dB)':<13} | {'Corr':<6}")
    print("-" * 80)
    for r in results:
        if "error" in r:
            print(f"{r.get('tier', 'Unknown'):<22} | Error: {r['error']}")
            continue
        print(
            f"{r['tier']:<22} | {r['size_mb']:<9.2f} | {r['per_frame_latency_ms']:<10.2f} | "
            f"{r['realtime_ratio']:<9.3f} | {r['snr_improvement_db']:<13.2f} | {r['speech_correlation']:<6.3f}"
        )
    print("=" * 80)

    report = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": sys.version,
        "sample_rate": sample_rate,
        "tiers": results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nReport written to: {out_file}\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Precision & Deployment Benchmark")
    parser.add_argument("--output", default="results/ai_deployment/ai_deployment_comparison.json", help="Path to save report JSON")
    args = parser.parse_args()
    run_deployment_benchmark(args.output)
