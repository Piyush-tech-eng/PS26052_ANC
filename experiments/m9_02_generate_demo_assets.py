"""M9.02 — Generate backup demo assets.

Produces:
(a) Pre-recorded before/after audio pairs for offline playback if the live
    rig fails.
(b) A one-page metrics summary formatted for a presentation slide.

Usage::

    python experiments/m9_02_generate_demo_assets.py \\
        --output results/demo_assets
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from anc.speech.sources import (
    generate_synthetic_speech,
    generate_synthetic_rotor,
    generate_synthetic_engine,
    generate_synthetic_impulsive,
    generate_synthetic_wind,
)
from anc.speech.mixing import mix_at_snr
from anc.evaluation.metrics import compute_si_snr, compute_stoi, compute_pesq_approx
from ai.models.spectral_gate import SpectralGateModel


def generate_demo_assets(output_dir: str | Path) -> None:
    """Generate pre-recorded demo assets and a metrics summary.

    Parameters
    ----------
    output_dir : path
        Output directory for demo assets.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    sr = 16_000
    duration = 3.0  # 3-second clips for demo

    # Generate speech samples
    speakers = [
        generate_synthetic_speech(sr, duration, speaker_id=f"demo_spk_{i}", seed=i * 10)
        for i in range(3)
    ]

    # Generate noise samples from different families
    noises = [
        ("rotor", generate_synthetic_rotor(sr, duration, seed=100)),
        ("engine", generate_synthetic_engine(sr, duration, seed=101)),
        ("impulsive", generate_synthetic_impulsive(sr, duration, seed=102)),
        ("wind", generate_synthetic_wind(sr, duration, seed=103)),
    ]

    # Initialize the enhancement model
    model = SpectralGateModel(sample_rate=sr)

    results_summary = []
    asset_count = 0

    for spk_idx, speech in enumerate(speakers):
        for noise_name, noise in noises:
            for snr_db in [0.0, 5.0, 10.0]:
                mix = mix_at_snr(speech, noise, snr_db)

                # Process through enhancement model
                model.reset()
                enhanced = model.enhance(mix.noisy_speech)

                # Compute metrics
                si_snr_before = compute_si_snr(mix.noisy_speech, mix.clean_speech)
                si_snr_after = compute_si_snr(enhanced, mix.clean_speech)
                stoi_before = compute_stoi(mix.noisy_speech, mix.clean_speech, sr)
                stoi_after = compute_stoi(enhanced, mix.clean_speech, sr)
                pesq_before = compute_pesq_approx(mix.noisy_speech, mix.clean_speech, sr)
                pesq_after = compute_pesq_approx(enhanced, mix.clean_speech, sr)

                result = {
                    "speaker": speech.speaker_id,
                    "noise_family": noise_name,
                    "snr_db": snr_db,
                    "si_snr_before": round(si_snr_before, 2),
                    "si_snr_after": round(si_snr_after, 2),
                    "si_snr_improvement": round(si_snr_after - si_snr_before, 2),
                    "stoi_before": round(stoi_before, 3),
                    "stoi_after": round(stoi_after, 3),
                    "pesq_before": round(pesq_before, 2),
                    "pesq_after": round(pesq_after, 2),
                }
                results_summary.append(result)

                # Save audio files (as .npz for now; convert to WAV if scipy.io available)
                prefix = f"demo_{spk_idx}_{noise_name}_snr{snr_db:+.0f}"
                np.savez_compressed(
                    output / f"{prefix}.npz",
                    clean=mix.clean_speech,
                    noisy=mix.noisy_speech,
                    enhanced=enhanced,
                    noise=mix.noise_component,
                    sample_rate=np.array(sr),
                )

                # Try to save as WAV
                try:
                    from scipy.io import wavfile
                    wav_dir = output / "wav"
                    wav_dir.mkdir(exist_ok=True)

                    for suffix, audio in [("noisy", mix.noisy_speech),
                                          ("enhanced", enhanced),
                                          ("clean", mix.clean_speech)]:
                        pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
                        wavfile.write(str(wav_dir / f"{prefix}_{suffix}.wav"), sr, pcm)
                except ImportError:
                    pass

                asset_count += 1

    # Write metrics summary JSON
    summary = {
        "model": "spectral_gate",
        "sample_rate": sr,
        "duration_per_clip": duration,
        "num_assets": asset_count,
        "results": results_summary,
    }
    (output / "demo_metrics_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    # Generate presentation-ready metrics table
    print(f"\n{'='*80}")
    print(f"DEMO ASSETS GENERATED — {asset_count} before/after pairs")
    print(f"{'='*80}")
    print(f"\n{'Speaker':<12} {'Noise':<12} {'SNR':>5} "
          f"{'SI-SNR Before':>14} {'SI-SNR After':>13} {'dSI-SNR':>9} "
          f"{'STOI Before':>12} {'STOI After':>11}")
    print("-" * 100)

    for r in results_summary:
        print(f"{r['speaker']:<12} {r['noise_family']:<12} {r['snr_db']:>5.0f} "
              f"{r['si_snr_before']:>14.2f} {r['si_snr_after']:>13.2f} "
              f"{r['si_snr_improvement']:>+9.2f} "
              f"{r['stoi_before']:>12.3f} {r['stoi_after']:>11.3f}")

    # Averages
    avg_improvement = np.mean([r["si_snr_improvement"] for r in results_summary])
    avg_stoi_before = np.mean([r["stoi_before"] for r in results_summary])
    avg_stoi_after = np.mean([r["stoi_after"] for r in results_summary])
    print("-" * 100)
    print(f"{'AVERAGE':<30} {'':>5} "
          f"{'':>14} {'':>13} {avg_improvement:>+9.2f} "
          f"{avg_stoi_before:>12.3f} {avg_stoi_after:>11.3f}")

    print(f"\nAssets saved to: {output}")
    print(f"Metrics summary: {output / 'demo_metrics_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demo assets")
    parser.add_argument(
        "--output", default="results/demo_assets",
        help="Output directory (default: results/demo_assets)",
    )
    args = parser.parse_args()
    generate_demo_assets(args.output)
