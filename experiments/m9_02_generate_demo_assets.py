"""M9.02 — Generate backup demo assets.

Produces:
(a) Pre-recorded before/after audio pairs for offline playback if the live
    rig fails.
(b) A one-page metrics summary formatted for a presentation slide.
(c) Optionally, a side-by-side comparison of multiple models (e.g.,
    classical SpectralGate vs neural DTLN).

Usage::

    # Auto-select best model
    python experiments/m9_02_generate_demo_assets.py \\
        --output results/demo_assets

    # Force a specific model
    python experiments/m9_02_generate_demo_assets.py \\
        --model dtln --output results/demo_assets_dtln

    # Run comparison (both classical and AI)
    python experiments/m9_02_generate_demo_assets.py \\
        --compare --output results/demo_comparison
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
from anc.evaluation.metrics import (
    compute_si_snr,
    compute_stoi_standard,
    compute_pesq_standard,
)
from ai.models import get_best_available_model


def _run_one_model(model, speakers, noises, sr, output, snr_levels):
    """Run demo asset generation for a single model, returning metrics."""
    results_summary = []
    asset_count = 0

    for spk_idx, speech in enumerate(speakers):
        for noise_name, noise in noises:
            for snr_db in snr_levels:
                mix = mix_at_snr(speech, noise, snr_db)

                # Process through enhancement model
                if hasattr(model, "reset"):
                    model.reset()
                enhanced = model.enhance(mix.noisy_speech)

                # Compute metrics (prefer standard when available)
                si_snr_before = compute_si_snr(mix.noisy_speech, mix.clean_speech)
                si_snr_after = compute_si_snr(enhanced, mix.clean_speech)

                stoi_before, stoi_src = compute_stoi_standard(
                    mix.noisy_speech, mix.clean_speech, sr
                )
                stoi_after, _ = compute_stoi_standard(
                    enhanced, mix.clean_speech, sr
                )

                pesq_before, pesq_src = compute_pesq_standard(
                    mix.noisy_speech, mix.clean_speech, sr
                )
                pesq_after, _ = compute_pesq_standard(
                    enhanced, mix.clean_speech, sr
                )

                result = {
                    "speaker": speech.speaker_id,
                    "noise_family": noise_name,
                    "snr_db": snr_db,
                    "model": model.name,
                    "si_snr_before": round(si_snr_before, 2),
                    "si_snr_after": round(si_snr_after, 2),
                    "si_snr_improvement": round(si_snr_after - si_snr_before, 2),
                    "stoi_before": round(stoi_before, 3),
                    "stoi_after": round(stoi_after, 3),
                    "stoi_source": stoi_src,
                    "pesq_before": round(pesq_before, 2),
                    "pesq_after": round(pesq_after, 2),
                    "pesq_source": pesq_src,
                }
                results_summary.append(result)

                # Save audio files
                prefix = f"demo_{spk_idx}_{noise_name}_snr{snr_db:+.0f}"
                np.savez_compressed(
                    output / f"{prefix}.npz",
                    clean=mix.clean_speech,
                    noisy=mix.noisy_speech,
                    enhanced=enhanced,
                    noise=mix.noise_component,
                    sample_rate=np.array(sr),
                    model_name=np.array(model.name),
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

    return results_summary, asset_count


def _print_summary(results_summary, model_name, asset_count):
    """Print a presentation-ready metrics table."""
    print(f"\n{'='*100}")
    print(f"DEMO ASSETS: {model_name.upper()} — {asset_count} pairs")
    print(f"{'='*100}")
    print(f"\n{'Speaker':<12} {'Noise':<12} {'SNR':>5} "
          f"{'SI-SNR Bef':>11} {'SI-SNR Aft':>11} {'dSI-SNR':>9} "
          f"{'STOI Bef':>9} {'STOI Aft':>9} "
          f"{'PESQ Bef':>9} {'PESQ Aft':>9}")
    print("-" * 100)

    for r in results_summary:
        print(f"{r['speaker']:<12} {r['noise_family']:<12} {r['snr_db']:>5.0f} "
              f"{r['si_snr_before']:>11.2f} {r['si_snr_after']:>11.2f} "
              f"{r['si_snr_improvement']:>+9.2f} "
              f"{r['stoi_before']:>9.3f} {r['stoi_after']:>9.3f} "
              f"{r['pesq_before']:>9.2f} {r['pesq_after']:>9.2f}")

    # Averages
    avg_improvement = np.mean([r["si_snr_improvement"] for r in results_summary])
    avg_stoi_before = np.mean([r["stoi_before"] for r in results_summary])
    avg_stoi_after = np.mean([r["stoi_after"] for r in results_summary])
    avg_pesq_before = np.mean([r["pesq_before"] for r in results_summary])
    avg_pesq_after = np.mean([r["pesq_after"] for r in results_summary])
    print("-" * 100)
    print(f"{'AVERAGE':<30} {'':<5} "
          f"{'':>11} {'':>11} {avg_improvement:>+9.2f} "
          f"{avg_stoi_before:>9.3f} {avg_stoi_after:>9.3f} "
          f"{avg_pesq_before:>9.2f} {avg_pesq_after:>9.2f}")

    # Report metric source
    stoi_src = results_summary[0].get("stoi_source", "unknown")
    pesq_src = results_summary[0].get("pesq_source", "unknown")
    print(f"\nMetric sources: STOI={stoi_src}, PESQ={pesq_src}")


def generate_demo_assets(
    output_dir: str | Path,
    model_name: str = "auto",
    compare: bool = False,
) -> None:
    """Generate pre-recorded demo assets and a metrics summary.

    Parameters
    ----------
    output_dir : path
        Output directory for demo assets.
    model_name : str
        Model to use: ``'auto'``, ``'dtln'``, ``'rnnoise'``,
        ``'spectral_gate'``.
    compare : bool
        If True, run both the best AI model and SpectralGate and produce
        a comparison JSON.
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

    snr_levels = [0.0, 5.0, 10.0]

    if compare:
        # Run both AI and classical, save side-by-side
        from ai.models.spectral_gate import SpectralGateModel

        ai_model = get_best_available_model(prefer=model_name, sample_rate=sr)
        sg_model = SpectralGateModel(sample_rate=sr)

        # AI model assets
        ai_out = output / f"ai_{ai_model.name}"
        ai_out.mkdir(parents=True, exist_ok=True)
        ai_results, ai_count = _run_one_model(
            ai_model, speakers, noises, sr, ai_out, snr_levels
        )
        _print_summary(ai_results, ai_model.name, ai_count)

        # Classical baseline
        sg_out = output / "classical_spectral_gate"
        sg_out.mkdir(parents=True, exist_ok=True)
        sg_results, sg_count = _run_one_model(
            sg_model, speakers, noises, sr, sg_out, snr_levels
        )
        _print_summary(sg_results, sg_model.name, sg_count)

        # Comparison summary
        comparison = {
            "ai_model": ai_model.name,
            "classical_model": sg_model.name,
            "sample_rate": sr,
            "duration_per_clip": duration,
            "ai_results": ai_results,
            "classical_results": sg_results,
            "ai_avg_si_snr_improvement": round(
                float(np.mean([r["si_snr_improvement"] for r in ai_results])), 2
            ),
            "classical_avg_si_snr_improvement": round(
                float(np.mean([r["si_snr_improvement"] for r in sg_results])), 2
            ),
        }
        (output / "comparison_summary.json").write_text(
            json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nComparison saved to: {output / 'comparison_summary.json'}")
    else:
        # Single model run
        model = get_best_available_model(prefer=model_name, sample_rate=sr)
        results, count = _run_one_model(
            model, speakers, noises, sr, output, snr_levels
        )
        _print_summary(results, model.name, count)

        # Write metrics summary JSON
        summary = {
            "model": model.name,
            "sample_rate": sr,
            "duration_per_clip": duration,
            "num_assets": count,
            "stoi_source": results[0].get("stoi_source", "unknown") if results else "unknown",
            "pesq_source": results[0].get("pesq_source", "unknown") if results else "unknown",
            "results": results,
        }
        (output / "demo_metrics_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\nAssets saved to: {output}")
        print(f"Metrics summary: {output / 'demo_metrics_summary.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate demo assets")
    parser.add_argument(
        "--output", default="results/demo_assets",
        help="Output directory (default: results/demo_assets)",
    )
    parser.add_argument(
        "--model", default="auto",
        choices=["auto", "dtln", "rnnoise", "spectral_gate"],
        help="Model to use (default: auto = best available)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Run both AI and classical models for side-by-side comparison",
    )
    args = parser.parse_args()
    generate_demo_assets(args.output, model_name=args.model, compare=args.compare)
