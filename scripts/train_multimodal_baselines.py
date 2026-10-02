from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bad_ads_regulation.classifier_data import ClassifierDataset, read_csv_flexible
from bad_ads_regulation.classifier_models import create_classifier
from bad_ads_regulation.classifier_training import (
    build_collate_fn,
    build_image_transform,
    positive_class_weight,
    run_epoch,
)

# keep repeated training runs reproducible
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# combine training and validation metrics for one epoch
def metric_record(epoch: int, train: dict, validation: dict) -> dict:
    record = {"epoch": epoch}
    record.update({f"train_{key}": value for key, value in train.items()})
    record.update({f"validation_{key}": value for key, value in validation.items()})
    return record


def main() -> None:
    # define model training and runtime settings
    parser = argparse.ArgumentParser(description="Train one weak-label multimodal baseline.")
    parser.add_argument("--model", choices=("text", "image", "fusion"), required=True)
    parser.add_argument("--manifest", default="outputs/multimodal_baselines/classifier_manifest_with_ocr.csv")
    parser.add_argument("--output-dir", default="outputs/multimodal_baselines/runs")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--unfreeze-backbone", action="store_true")
    args = parser.parse_args()

    # reject invalid settings before loading data or models
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must be positive.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    # set the seed choose a device and load the manifest
    set_seed(args.seed)
    if args.device == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device_name = args.device
    device = torch.device(device_name)
    manifest = read_csv_flexible(args.manifest)
    # check that the manifest has the required fields and splits
    required_columns = {"ad_id", "screenshot_path", "label", "split", "ocr_text"}
    missing = required_columns - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if set(manifest["split"]) != {"train", "validation", "test"}:
        raise ValueError("Manifest must contain train, validation, and test splits.")

    # build training and validation data loaders
    train_frame = manifest.loc[manifest["split"] == "train"].copy()
    validation_frame = manifest.loc[manifest["split"] == "validation"].copy()
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    collate_fn = build_collate_fn(tokenizer, max_length=args.max_length)
    transform = build_image_transform()
    train_loader = DataLoader(
        ClassifierDataset(train_frame, transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    validation_loader = DataLoader(
        ClassifierDataset(validation_frame, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # create the model optimizer and class-weighted loss
    model = create_classifier(args.model, freeze_backbone=not args.unfreeze_backbone).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate)
    weight = torch.tensor([positive_class_weight(train_frame["label"].tolist())], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=weight)

    # prepare the output folder and best-checkpoint tracking
    run_dir = Path(args.output_dir) / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_f1 = float("-inf")
    best_epoch = 0
    checkpoint_path = run_dir / "best_checkpoint.pt"

    # train and validate the model once per epoch
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, args.model, train_loader, device, loss_fn, optimizer=optimizer)
        validation_metrics = run_epoch(model, args.model, validation_loader, device, loss_fn)
        history.append(metric_record(epoch, train_metrics, validation_metrics))
        print(
            f"epoch={epoch} train_loss={train_metrics['loss']:.4f} train_f1={train_metrics['f1']:.4f} "
            f"validation_loss={validation_metrics['loss']:.4f} validation_f1={validation_metrics['f1']:.4f}"
        )
        # save a checkpoint whenever validation f1 improves
        if validation_metrics["f1"] > best_f1:
            best_f1 = float(validation_metrics["f1"])
            best_epoch = epoch
            torch.save(
                {
                    "model_name": args.model,
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                    "max_length": args.max_length,
                    "freeze_backbone": not args.unfreeze_backbone,
                    "seed": args.seed,
                },
                checkpoint_path,
            )

    # save the settings results and full training history
    summary = {
        "model": args.model,
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_length": args.max_length,
        "freeze_backbone": not args.unfreeze_backbone,
        "train_examples": len(train_frame),
        "validation_examples": len(validation_frame),
        "test_examples_unused": int((manifest["split"] == "test").sum()),
        "positive_class_weight_from_train": float(weight.item()),
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "checkpoint": str(checkpoint_path),
        "history": history,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Saved best checkpoint from epoch {best_epoch} to {checkpoint_path}")
    # print(f"Saved training summary to {run_dir / 'training_summary.json'}")



if __name__ == "__main__":
    main()
