from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from torchvision import models


def _weights_arg(builder_name: str, pretrained: bool):
    if not pretrained:
        return None
    return "DEFAULT"


def _replace_classifier(model: nn.Module, num_classes: int, dropout: float = 0.2) -> nn.Module:
    """Replace the classification head of common torchvision models."""
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        return model

    if hasattr(model, "classifier"):
        classifier = model.classifier
        if isinstance(classifier, nn.Linear):
            model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(classifier.in_features, num_classes))
            return model

        if isinstance(classifier, nn.Sequential):
            for idx in reversed(range(len(classifier))):
                if isinstance(classifier[idx], nn.Linear):
                    in_features = classifier[idx].in_features
                    classifier[idx] = nn.Linear(in_features, num_classes)
                    model.classifier = classifier
                    return model

    raise ValueError(f"Unsupported architecture head for {model.__class__.__name__}")


class CNNTransformerHead(nn.Module):
    """Small experimental CNN + Transformer classifier.

    This module is included to document the hybrid direction explored during the
    experiments. It uses a CNN backbone as feature extractor and a lightweight
    Transformer encoder over flattened spatial tokens.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int, nhead: int = 8, layers: int = 2):
        super().__init__()
        self.backbone = backbone
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=nhead, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.ndim == 4:
            b, c, h, w = features.shape
            features = features.flatten(2).transpose(1, 2)
        features = self.transformer(features)
        pooled = features.mean(dim=1)
        return self.classifier(pooled)


def build_model(model_name: str, num_classes: int, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    name = model_name.lower().replace("-", "_")
    builders: dict[str, Callable[..., nn.Module]] = {
        "resnet18": models.resnet18,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "densenet121": models.densenet121,
        "densenet201": models.densenet201,
        "mobilenet_v3_large": models.mobilenet_v3_large,
        "mobilenetv3": models.mobilenet_v3_large,
        "efficientnet_b0": models.efficientnet_b0,
        "efficientnet_b3": models.efficientnet_b3,
        "efficientnet_b7": models.efficientnet_b7,
    }
    if name not in builders:
        raise ValueError(f"Unsupported model '{model_name}'. Available: {sorted(builders)}")

    model = builders[name](weights=_weights_arg(name, pretrained))
    return _replace_classifier(model, num_classes=num_classes, dropout=dropout)


def load_local_pretrained_weights(model: nn.Module, weights_path: str | None) -> nn.Module:
    """Load local weights when working on an offline HPC node.

    Some clusters do not allow direct internet access from compute nodes. In that
    case, weights can be downloaded once, copied to the cluster and loaded here.
    """
    if not weights_path:
        return model
    state = torch.load(weights_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=False)
    return model
