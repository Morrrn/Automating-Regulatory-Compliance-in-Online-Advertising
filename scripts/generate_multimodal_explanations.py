from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer

from bad_ads_regulation.classifier_data import read_csv_flexible
from bad_ads_regulation.classifier_explanations import gradcam_for_image, select_explanation_cases, text_gradient_attributions
from bad_ads_regulation.classifier_models import create_classifier
from bad_ads_regulation.classifier_training import build_image_transform


def save_overlay(image: Image.Image, cam: torch.Tensor, path: Path) -> None:
    # blend the original image with a resized grad-cam heatmap
    base = image.convert("RGB")
    values = np.array(Image.fromarray((cam.cpu().numpy() * 255).astype(np.uint8)).resize(base.size), dtype=np.float32) / 255
    heat = np.stack([np.ones_like(values), values, np.zeros_like(values)], axis=-1) * 255
    overlay = Image.fromarray(heat.astype(np.uint8), "RGB")
    Image.blend(base, overlay, 0.45).save(path)


def main() -> None:
    # define input folders and the number of cases per group
    parser = argparse.ArgumentParser(description="Generate local Grad-CAM and token attribution artefacts.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--evaluation-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--per-group", type=int, default=2)
    args = parser.parse_args()
    # choose a device and load the held-out test rows
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest = read_csv_flexible(args.manifest)
    test_rows = manifest.loc[manifest["split"] == "test"].copy()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    # prepare the shared tokenizer and image transform
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    transform = build_image_transform()
    selected_all = []
    # select representative cases for each model
    for model_name in ("text", "image", "fusion"):
        predictions = read_csv_flexible(Path(args.evaluation_dir) / f"{model_name}_test_predictions.csv")
        selected = select_explanation_cases(predictions, model_name, args.per_group).merge(
            test_rows[["ad_id", "screenshot_path", "ocr_text"]].assign(ad_id=lambda x: x.ad_id.astype(str)), on="ad_id", how="left", validate="one_to_one"
        )
        if selected["screenshot_path"].isna().any(): raise RuntimeError("Selected ID missing from test manifest.")
        selected_all.append(selected)
        # load the matching trained checkpoint
        checkpoint = torch.load(Path(args.run_dir) / model_name / "best_checkpoint.pt", map_location=device, weights_only=True)
        model = create_classifier(model_name, freeze_backbone=bool(checkpoint["freeze_backbone"])).to(device)
        model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        # create a separate output folder for each selected case
        for _, row in selected.iterrows():
            case_dir = out / model_name / str(row.ad_id); case_dir.mkdir(parents=True, exist_ok=True)
            original = Image.open(row.screenshot_path).convert("RGB")
            image = transform(original).unsqueeze(0).to(device)
            predicted_positive = bool(row.prediction)
            metadata = {key: (float(row[key]) if key in {"probability", "threshold_distance"} 
                              else str(row[key])) for key in ["ad_id", "model", "weak_label", "prediction", 
                                                              "probability", "confusion_group", "selection_rank", "threshold_distance"]}
            # create grad-cam overlays for models that use images
            if model_name in {"image", "fusion"}:
                if model_name == "image":
                    cam = gradcam_for_image(model, model.backbone.layer4, lambda: model(image), predicted_positive)
                else:
                    encoded = tokenizer(str(row.ocr_text), return_tensors="pt", truncation=True, max_length=128).to(device)
                    cam = gradcam_for_image(model, model.image_encoder.backbone.layer4, 
                                            lambda: model(encoded["input_ids"], encoded["attention_mask"], image), predicted_positive)
                save_overlay(original, cam, case_dir / "gradcam_overlay.png")
            # create token attribution tables for models that use text
            if model_name in {"text", "fusion"}:
                if str(row.ocr_text).strip():
                    if model_name == "text": score_fn = lambda embedding: model.classifier(embedding)
                    else:
                        image_embedding = model.image_encoder.encode(image).detach()
                        score_fn = lambda embedding: model.classifier(torch.cat([embedding, image_embedding], dim=1))
                    tokens = text_gradient_attributions(model if model_name == "text" 
                                                        else model.text_encoder, tokenizer, str(row.ocr_text), device, score_fn, predicted_positive)
                    pd.DataFrame(tokens).to_csv(case_dir / "token_attributions.csv", index=False, encoding="utf-8-sig")
                else: pd.DataFrame(columns=["token", "attribution"]).to_csv(case_dir / "token_attributions.csv", index=False, encoding="utf-8-sig")
            # save metadata for each selected case
            (case_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # save the full case list
    pd.concat(selected_all, ignore_index=True).to_csv(out / "selected_cases.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {sum(len(frame) for frame in selected_all)} selected cases to {out}")


if __name__ == "__main__": main()
