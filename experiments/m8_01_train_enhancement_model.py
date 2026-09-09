"""M8.01 — Train speech enhancement model.

Runnable training experiment entry point.  Trains either the fine-tuned
DTLN or from-scratch Conv-TasNet on the Phase A dataset, writing
checkpoints and training curves to ``results/m8_01_train_enhancement_model/``.

Usage::

    python experiments/m8_01_train_enhancement_model.py \\
        --dataset results/m7_speech_ai_handoff \\
        --model dtln \\
        --output results/m8_01_train_enhancement_model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a speech enhancement model."
    )
    parser.add_argument(
        "--dataset", type=str,
        default="results/m7_speech_ai_handoff",
        help="Path to the generated dataset directory.",
    )
    parser.add_argument(
        "--model", type=str, default="dtln",
        choices=["dtln", "conv_tasnet"],
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--output", type=str,
        default="results/m8_01_train_enhancement_model",
        help="Output directory for checkpoints and logs.",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--max-epochs", type=int, default=200, help="Max epochs.")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda).")
    parser.add_argument(
        "--from-onnx", action="store_true",
        help="Initialize DTLN from existing ONNX weights (fine-tuning).",
    )

    args = parser.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required for training.")
        print("  pip install torch torchaudio")
        sys.exit(1)

    from ai.training.dataloader import create_dataloaders
    from ai.training.trainer import TrainingConfig, train_model

    # Create dataloaders
    print(f"Loading dataset from: {args.dataset}")
    loaders = create_dataloaders(
        args.dataset,
        batch_size=args.batch_size,
        augment_train=True,
    )

    if "train" not in loaders or "validation" not in loaders:
        print("ERROR: Dataset must have train and validation splits.")
        sys.exit(1)

    # Create model
    print(f"Creating model: {args.model}")
    if args.model == "dtln":
        from ai.models.dtln_trainable import TrainableDTLN, load_from_onnx

        if args.from_onnx:
            model_dir = Path("models/dtln")
            model = load_from_onnx(
                model_dir / "model_1.onnx",
                model_dir / "model_2.onnx",
            )
            print("  Initialized from ONNX pretrained weights (fine-tuning mode).")
        else:
            model = TrainableDTLN()
            print("  Random initialization (from-scratch mode).")
    elif args.model == "conv_tasnet":
        from ai.models.conv_tasnet import ConvTasNet
        model = ConvTasNet()
        print("  Random initialization (from-scratch mode).")
    else:
        print(f"Unknown model: {args.model}")
        sys.exit(1)

    # Training config
    config = TrainingConfig(
        learning_rate=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )

    # Train
    output_dir = Path(args.output) / args.model
    results = train_model(
        model,
        loaders["train"],
        loaders["validation"],
        config,
        output_dir,
        device=args.device,
    )

    import shutil
    best_ckpt = output_dir / "best_model.pt"
    if best_ckpt.exists() and args.model == "dtln":
        target_dir = Path("models/dtln_finetuned")
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_ckpt, target_dir / "best_model.pt")
        print(f"Copied best model to {target_dir / 'best_model.pt'}")

    print(f"\nResults: {results}")


if __name__ == "__main__":
    main()
