from __future__ import annotations

from collections.abc import Callable, Sequence
import torch
from torchvision import transforms
from bad_ads_regulation.classifier_metrics import binary_metrics, prediction_table


def build_image_transform():
    # resize and normalize images for the resnet backbone
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def positive_class_weight(labels: Sequence[int]) -> float:
    # give positive cases more weight when negative labels are more common
    positives = sum(int(label) for label in labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training labels must contain both classes.")
    return negatives / positives


# tokenize OCR text while retaining its pairing with screenshot tensors
def build_collate_fn(tokenizer, max_length: int) -> Callable[[Sequence[dict[str, object]]], dict[str, object]]:

    def collate(items: Sequence[dict[str, object]]) -> dict[str, object]:
        # tokenize text and stack its paired images and labels
        texts = [str(item["text"]) for item in items]
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        return {
            "ad_ids": [str(item["ad_id"]) for item in items],
            "images": torch.stack([item["image"] for item in items]),
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": torch.stack([item["label"] for item in items]).float(),
        }

    return collate


def forward_logits(model: torch.nn.Module, model_name: str, batch: dict[str, object]) -> torch.Tensor:
    # send the fields required by the selected model
    if model_name == "text":
        return model(batch["input_ids"], batch["attention_mask"])
    if model_name == "image":
        return model(batch["images"])
    if model_name == "fusion":
        return model(batch["input_ids"], batch["attention_mask"], batch["images"])
    raise ValueError(f"Unknown model name: {model_name}")


def move_batch_to_device(batch: dict[str, object], device: torch.device) -> dict[str, object]:
    # move tensor fields while keeping ad ids on the cpu
    moved = dict(batch)
    for key in ("images", "input_ids", "attention_mask", "labels"):
        moved[key] = batch[key].to(device)
    return moved


def run_epoch(
    model: torch.nn.Module,
    model_name: str,
    batches,
    device: torch.device,
    loss_fn,
    optimizer=None,
    forward_fn=forward_logits,
) -> dict[str, float | int]:
    """Run a train or validation epoch without exposing held-out test data."""
    # use the optimizer to switch between training and validation modes
    is_training = optimizer is not None
    model.train(is_training)
    targets: list[int] = []
    probabilities: list[float] = []
    total_loss = 0.0
    total_examples = 0

    # enable gradients and weight updates only during training
    with torch.set_grad_enabled(is_training):
        for batch in batches:
            batch = move_batch_to_device(batch, device)
            logits = forward_fn(model, model_name, batch)
            labels = batch["labels"]
            loss = loss_fn(logits, labels)
            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_examples += len(labels)
            total_loss += float(loss.detach().item()) * len(labels)
            targets.extend(labels.detach().round().to(torch.int64).cpu().tolist())
            probabilities.extend(torch.sigmoid(logits.detach()).cpu().tolist())

    # use one fixed threshold for comparable model metrics
    predictions = [int(probability >= 0.5) for probability in probabilities]
    metrics = binary_metrics(targets, predictions, probabilities)
    metrics["loss"] = total_loss / total_examples if total_examples else float("nan")
    return metrics


def collect_predictions(model: torch.nn.Module, model_name: str, batches, device: torch.device):
    # run inference without gradients and keep predictions aligned with ad ids
    model.eval()
    ad_ids: list[str] = []
    targets: list[int] = []
    probabilities: list[float] = []
    with torch.no_grad():
        for batch in batches:
            ad_ids.extend(batch["ad_ids"])
            moved = move_batch_to_device(batch, device)
            logits = forward_logits(model, model_name, moved)
            targets.extend(moved["labels"].round().to(torch.int64).cpu().tolist())
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
    return prediction_table(ad_ids, targets, probabilities)
