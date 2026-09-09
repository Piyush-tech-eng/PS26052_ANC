"""M8.02 — Ablation study: loss term combinations.

Reruns the training pipeline with different loss-term subsets to determine
which combination produces the best STOI/PESQ.

Configurations tested:
  1. SI-SNR only
  2. SI-SNR + L1
  3. SI-SNR + perceptual STFT
  4. SI-SNR + L1 + perceptual STFT (full combined)

Usage::

    python experiments/m8_02_ablation_study.py \\
        --dataset results/m7_speech_ai_handoff \\
        --output results/m8_02_ablation_study
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation study: loss terms.")
    parser.add_argument("--dataset", default="results/m7_speech_ai_handoff")
    parser.add_argument("--output", default="results/m8_02_ablation_study")
    parser.add_argument("--model", default="dtln", choices=["dtln", "conv_tasnet"])
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--from-onnx", action="store_true", help="Initialize DTLN from ONNX weights.")

    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required for the ablation study.")
        sys.exit(1)

    from ai.training.dataloader import create_dataloaders
    from ai.training.trainer import TrainingConfig, train_model

    # Define ablation configurations
    ablation_configs = [
        {
            "name": "si_snr_only",
            "si_snr_weight": 1.0,
            "l1_weight": 0.0,
            "stft_weight": 0.0,
        },
        {
            "name": "si_snr_l1",
            "si_snr_weight": 1.0,
            "l1_weight": 0.1,
            "stft_weight": 0.0,
        },
        {
            "name": "si_snr_stft",
            "si_snr_weight": 1.0,
            "l1_weight": 0.0,
            "stft_weight": 0.5,
        },
        {
            "name": "full_combined",
            "si_snr_weight": 1.0,
            "l1_weight": 0.1,
            "stft_weight": 0.5,
        },
    ]

    loaders = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        augment_train=True,
    )

    if "train" not in loaders or "validation" not in loaders:
        print("ERROR: Dataset must have train and validation splits.")
        sys.exit(1)

    output_path = Path(args.output)
    results: list[dict] = []

    for config_def in ablation_configs:
        run_name = config_def["name"]
        print(f"\n{'='*60}")
        print(f"Ablation: {run_name}")
        print(f"{'='*60}")

        # Create fresh model
        if args.model == "dtln":
            from ai.models.dtln_trainable import TrainableDTLN, load_from_onnx
            if args.from_onnx:
                model_dir = Path("models/dtln")
                model = load_from_onnx(model_dir / "model_1.onnx", model_dir / "model_2.onnx")
            else:
                model = TrainableDTLN()
        else:
            from ai.models.conv_tasnet import ConvTasNet
            model = ConvTasNet()

        config = TrainingConfig(
            learning_rate=1e-4,
            si_snr_weight=config_def["si_snr_weight"],
            l1_weight=config_def["l1_weight"],
            stft_weight=config_def["stft_weight"],
            max_epochs=args.max_epochs,
            patience=10,
        )

        try:
            run_result = train_model(
                model, loaders["train"], loaders["validation"],
                config, output_path / run_name,
                device=args.device,
            )

            results.append({
                "name": run_name,
                "si_snr_weight": config_def["si_snr_weight"],
                "l1_weight": config_def["l1_weight"],
                "stft_weight": config_def["stft_weight"],
                "best_stoi": run_result["best_stoi"],
                "best_epoch": run_result["best_epoch"],
                "total_epochs": run_result["total_epochs"],
            })
        except Exception as exc:
            results.append({
                "name": run_name,
                "error": str(exc),
            })
            print(f"  [FAIL] Failed: {exc}")

    # Write ablation table
    output_path.mkdir(parents=True, exist_ok=True)

    csv_path = output_path / "ablation_table.csv"
    if results:
        fieldnames = list(results[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    json_path = output_path / "ablation_results.json"
    json_path.write_text(
        json.dumps(results, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    print(f"\n{'='*60}")
    print("Ablation study complete.")
    print(f"Results: {csv_path}")
    print(f"\nSummary:")
    for r in sorted(results, key=lambda x: x.get("best_stoi", -1), reverse=True):
        stoi = r.get("best_stoi", "N/A")
        print(f"  {r['name']:>20s}: STOI = {stoi}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
