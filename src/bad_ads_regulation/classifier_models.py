from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from transformers import AutoModel


class ImageClassifier(nn.Module):
    # use the same compact embedding size as the other baselines
    embedding_dim = 128

    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True) -> None:
        super().__init__()
        # load resnet and optionally train only its final feature block
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.backbone.layer4.parameters():
                parameter.requires_grad = True
        # map image features to one binary classification score
        self.projection = nn.Sequential(nn.Linear(512, self.embedding_dim), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Linear(self.embedding_dim, 1)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(images))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(images)).squeeze(-1)


class TextClassifier(nn.Module):
    # use the same compact embedding size as the image baseline
    embedding_dim = 128

    def __init__(self, model_name: str = "distilbert-base-uncased", freeze_backbone: bool = True) -> None:
        super().__init__()
        # load the text backbone and optionally train only its final layer
        self.backbone = AutoModel.from_pretrained(model_name)
        if freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            for parameter in self.backbone.transformer.layer[-1].parameters():
                parameter.requires_grad = True
        self.projection = nn.Sequential(
            nn.Linear(self.backbone.config.hidden_size, self.embedding_dim), nn.ReLU(), nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(self.embedding_dim, 1)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        output = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        # average visible token states into one text embedding
        pooled = (output.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.projection(pooled)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(input_ids, attention_mask)).squeeze(-1)


class LateFusionClassifier(nn.Module):
    def __init__(self, text_encoder: nn.Module, image_encoder: nn.Module) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        # combine separate text and image embeddings before classification
        fusion_dim = text_encoder.embedding_dim + image_encoder.embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        text_embedding = self.text_encoder.encode(input_ids, attention_mask)
        image_embedding = self.image_encoder.encode(images)
        return self.classifier(torch.cat([text_embedding, image_embedding], dim=1)).squeeze(-1)


def create_classifier(
    model_name: str,
    freeze_backbone: bool = True,
    pretrained: bool = True,
    text_model_name: str = "distilbert-base-uncased",
) -> nn.Module:
    # build the selected baseline with shared backbone settings
    if model_name == "text":
        return TextClassifier(model_name=text_model_name, freeze_backbone=freeze_backbone)
    if model_name == "image":
        return ImageClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone)
    if model_name == "fusion":
        return LateFusionClassifier(
            text_encoder=TextClassifier(model_name=text_model_name, freeze_backbone=freeze_backbone),
            image_encoder=ImageClassifier(pretrained=pretrained, freeze_backbone=freeze_backbone),
        )
    raise ValueError(f"Unknown model name: {model_name}")
