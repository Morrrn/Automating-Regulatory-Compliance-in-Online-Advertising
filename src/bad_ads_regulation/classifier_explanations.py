from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def normalise_cam(cam: torch.Tensor) -> torch.Tensor:
    # scale non-negative heatmap values to the zero-to-one range
    cam = cam.detach().float().clamp(min=0)
    cam = cam - cam.min()
    maximum = cam.max()
    return cam / maximum if maximum > 0 else torch.zeros_like(cam)


def merge_wordpieces(tokens: list[str], attributions: list[float]) -> list[dict[str, float]]:
    # join subword tokens and add their attribution scores
    if len(tokens) != len(attributions):
        raise ValueError("tokens and attributions must have equal length.")
    merged: list[dict[str, float]] = []
    special_tokens = {"[CLS]", "[SEP]", "[PAD]"}
    for token, attribution in zip(tokens, attributions, strict=True):
        if token in special_tokens:
            continue
        if token.startswith("##") and merged:
            merged[-1]["token"] += token[2:]
            merged[-1]["attribution"] += float(attribution)
        else:
            merged.append({"token": token, "attribution": float(attribution)})
    return merged


def gradcam_for_image(model, layer, forward_logit, predicted_positive: bool) -> torch.Tensor:
    # use grad-cam only as a diagnostic view of the trained classifier
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    # capture activations and gradients with temporary hooks
    forward_handle = layer.register_forward_hook(lambda _module, _inputs, output: activations.append(output))
    backward_handle = layer.register_full_backward_hook(
        lambda _module, _grad_inputs, grad_outputs: gradients.append(grad_outputs[0])
    )
    try:
        # backpropagate the score for the predicted class
        model.zero_grad(set_to_none=True)
        logit = forward_logit().reshape(-1)[0]
        (logit if predicted_positive else -logit).backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hook did not receive activations and gradients.")
        # combine feature channels and scale the final heatmap
        weights = gradients[-1].mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations[-1]).sum(dim=1)[0]
        return normalise_cam(cam.relu())
    # always remove hooks after the diagnostic pass
    finally:
        forward_handle.remove()
        backward_handle.remove()


def text_gradient_attributions(text_encoder, tokenizer, text: str, device: torch.device, score_fn, predicted_positive: bool):
    # tokenize the text and track gradients on its embeddings
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    embeddings = text_encoder.backbone.embeddings(input_ids).detach().requires_grad_(True)
    text_encoder.zero_grad(set_to_none=True)
    # pool visible token states and calculate the selected class score
    output = text_encoder.backbone(inputs_embeds=embeddings, attention_mask=attention_mask)
    mask = attention_mask.unsqueeze(-1).float()
    pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    projected = text_encoder.projection(pooled)
    logit = score_fn(projected).reshape(-1)[0]
    (logit if predicted_positive else -logit).backward()
    # combine gradients and embeddings into token scores
    scores = (embeddings.grad * embeddings).sum(dim=-1)[0].detach().cpu().tolist()
    return merge_wordpieces(tokenizer.convert_ids_to_tokens(input_ids[0]), scores)


def select_explanation_cases(predictions: pd.DataFrame, model_name: str, per_group: int = 2) -> pd.DataFrame:
    # choose cases closest to the threshold for error analysis
    required = {"ad_id", "weak_label", "prediction", "probability"}
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Prediction table is missing columns: {sorted(missing)}")
    if per_group < 1:
        raise ValueError("per_group must be positive.")

    # assign each prediction to a confusion group
    result = predictions.copy()
    result["ad_id"] = result["ad_id"].astype(str)
    result["confusion_group"] = np.select(
        [
            (result["weak_label"] == 1) & (result["prediction"] == 1),
            (result["weak_label"] == 0) & (result["prediction"] == 0),
            (result["weak_label"] == 0) & (result["prediction"] == 1),
            (result["weak_label"] == 1) & (result["prediction"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
        default="",
    )
    # require enough cases in every group
    expected_groups = {"TP", "TN", "FP", "FN"}
    counts = result["confusion_group"].value_counts().to_dict()
    missing_groups = sorted(group for group in expected_groups if counts.get(group, 0) < per_group)
    if missing_groups:
        raise ValueError(f"Insufficient cases for confusion groups: {', '.join(missing_groups)}")

    # rank each group by distance from the decision threshold
    result["threshold_distance"] = (result["probability"].astype(float) - 0.5).abs()
    selected = (
        result.sort_values(["confusion_group", "threshold_distance", "ad_id"])
        .groupby("confusion_group", group_keys=False, sort=True)
        .head(per_group)
        .copy()
    )
    selected["model"] = model_name
    selected["selection_rank"] = selected.groupby("confusion_group", sort=False).cumcount() + 1
    return selected.reset_index(drop=True)
