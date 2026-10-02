from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from bad_ads_regulation.classifier_data import ClassifierDataset, read_csv_flexible
from bad_ads_regulation.classifier_metrics import binary_metrics
from bad_ads_regulation.classifier_models import create_classifier
from bad_ads_regulation.classifier_training import build_collate_fn, build_image_transform, collect_predictions


def main() -> None:
    # define evaluation inputs models and runtime settings
    parser = argparse.ArgumentParser(description="Evaluate saved multimodal baseline checkpoints once on the held-out test split.")
    parser.add_argument("--manifest", default="outputs/multimodal_baselines/classifier_manifest_with_ocr.csv")
    parser.add_argument("--run-dir", default="outputs/multimodal_baselines/runs")
    parser.add_argument("--output-dir", default="outputs/multimodal_baselines/evaluation")
    parser.add_argument("--models", nargs="+", choices=("text", "image", "fusion"), default=("text", "image", "fusion"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    # choose an available device and load the held-out test split
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device
    device = torch.device(device_name)
    manifest = read_csv_flexible(args.manifest)
    test_frame = manifest.loc[manifest["split"] == "test"].copy()
    # check that the test split is present and has both classes
    if len(test_frame) == 0:
        raise ValueError("Manifest has no held-out test rows.")
    if test_frame["label"].nunique() != 2:
        raise ValueError("Held-out test split must contain both weak-label classes.")

    # load the text tokenizer and prepare the output folder
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison: list[dict] = []

    # evaluate each requested model from its best saved checkpoint
    for model_name in args.models:
        checkpoint_path = Path(args.run_dir) / model_name / "best_checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Missing best checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if checkpoint["model_name"] != model_name:
            raise ValueError(f"Checkpoint model mismatch in {checkpoint_path}")

        # rebuild the test data pipeline with the saved text length
        transform = build_image_transform()
        loader = DataLoader(
            ClassifierDataset(test_frame, transform),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=build_collate_fn(tokenizer, max_length=int(checkpoint["max_length"])),
        )
        # restore the model and collect test predictions
        model = create_classifier(model_name, freeze_backbone=bool(checkpoint["freeze_backbone"])).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        predictions = collect_predictions(model, model_name, loader, device)
        metrics = binary_metrics(
            predictions["weak_label"].tolist(),
            predictions["prediction"].tolist(),
            predictions["probability"].tolist(),
        )

        # save predictions metrics and the confusion matrix
        predictions.to_csv(output_dir / f"{model_name}_test_predictions.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(
            [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]],
            index=["actual_negative", "actual_positive"],
            columns=["predicted_negative", "predicted_positive"],
        ).to_csv(output_dir / f"{model_name}_test_confusion_matrix.csv", encoding="utf-8-sig")
        model_result = {
            "model": model_name,
            "best_validation_epoch": int(checkpoint["epoch"]),
            "test_split": "held_out_once",
            **metrics,
        }
        (output_dir / f"{model_name}_test_metrics.json").write_text(
            json.dumps(model_result, indent=2, allow_nan=False), encoding="utf-8"
        )
        comparison.append(model_result)
        print(f"{model_name}: test_f1={metrics['f1']:.4f} test_auroc={metrics['auroc']:.4f}")

    # compare all evaluated models in one csv file
    pd.DataFrame(comparison).to_csv(output_dir / "test_metrics_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote held-out comparison to {output_dir / 'test_metrics_comparison.csv'}")


if __name__ == "__main__":
    main()
