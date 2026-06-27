from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torchvision import models


class EfficientNetTransformerClassifier(nn.Module):
    """Hybrid CNN-Transformer classifier.

    This model uses an EfficientNetV2 backbone as a convolutional feature
    extractor and then applies a Transformer encoder over spatial feature tokens.

    The idea behind this prototype was to compare a pure CNN approach against
    a hybrid CNN + self-attention design. The CNN extracts local visual patterns,
    while the Transformer block tries to model more global relations between
    different regions of the image.

    This architecture is especially interesting for medical image classification
    because some findings can depend on local texture patterns, while others may
    require a more global view of the radiograph.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        backbone_out_channels: int,
        grid_size: int = 7,
        embed_dim: int = 256,
        num_heads: int = 8,
        transformer_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.features = backbone.features
        self.grid_size = grid_size
        self.embed_dim = embed_dim

        self.pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.projection = nn.Conv2d(backbone_out_channels, embed_dim, kernel_size=1)

        num_tokens = grid_size * grid_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, num_tokens + 1, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=transformer_layers,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        self._init_transformer_parameters()

    def _init_transformer_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)

        for module in self.classifier:
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(x)
        feature_map = self.pool(feature_map)
        tokens = self.projection(feature_map)

        tokens = tokens.flatten(2).transpose(1, 2)

        batch_size = tokens.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.positional_embedding

        encoded = self.transformer(tokens)
        cls_representation = self.norm(encoded[:, 0])

        logits = self.classifier(cls_representation)
        return logits


def _get_torchvision_weights(weights_name: str, pretrained: bool):
    """Return torchvision weights in a version-tolerant way."""
    if not pretrained:
        return None

    weights_enum = getattr(models, weights_name, None)

    if weights_enum is None:
        return None

    return weights_enum.DEFAULT


def _replace_linear_head(
    model: nn.Module,
    in_features: int,
    num_classes: int,
    dropout: float,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )


def _infer_backbone_out_channels(features: nn.Module, image_size: int = 224) -> int:
    """Infer the number of channels produced by a CNN feature extractor."""
    was_training = features.training
    features.eval()

    with torch.no_grad():
        dummy = torch.zeros(1, 3, image_size, image_size)
        output = features(dummy)

    if was_training:
        features.train()

    if output.ndim != 4:
        raise ValueError(
            "Expected CNN feature extractor to return a 4D feature map, "
            f"but got shape {tuple(output.shape)}."
        )

    return int(output.shape[1])


def build_resnet18(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = _get_torchvision_weights("ResNet18_Weights", pretrained)
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = _replace_linear_head(in_features, num_classes, dropout)
    return model


def build_resnet50(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = _get_torchvision_weights("ResNet50_Weights", pretrained)
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = _replace_linear_head(in_features, num_classes, dropout)
    return model


def build_densenet121(num_classes: int, pretrained: bool, dropout: float) -> nn.Module:
    weights = _get_torchvision_weights("DenseNet121_Weights", pretrained)
    model = models.densenet121(weights=weights)
    in_features = model.classifier.in_features
    model.classifier = _replace_linear_head(in_features, num_classes, dropout)
    return model


def build_mobilenet_v3_large(
    num_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    weights = _get_torchvision_weights("MobileNet_V3_Large_Weights", pretrained)
    model = models.mobilenet_v3_large(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    if len(model.classifier) >= 3 and isinstance(model.classifier[2], nn.Dropout):
        model.classifier[2].p = dropout

    return model


def build_efficientnet_b7(
    num_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    weights = _get_torchvision_weights("EfficientNet_B7_Weights", pretrained)
    model = models.efficientnet_b7(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = _replace_linear_head(in_features, num_classes, dropout)
    return model


def build_efficientnet_v2_s(
    num_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    weights = _get_torchvision_weights("EfficientNet_V2_S_Weights", pretrained)
    model = models.efficientnet_v2_s(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = _replace_linear_head(in_features, num_classes, dropout)
    return model


def build_efficientnet_v2_m(
    num_classes: int,
    pretrained: bool,
    dropout: float,
) -> nn.Module:
    weights = _get_torchvision_weights("EfficientNet_V2_M_Weights", pretrained)
    model = models.efficientnet_v2_m(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier = _replace_linear_head(in_features, num_classes, dropout)
    return model


def build_efficientnet_v2_transformer(
    num_classes: int,
    pretrained: bool,
    dropout: float,
    embed_dim: int = 256,
    num_heads: int = 8,
    transformer_layers: int = 2,
    grid_size: int = 7,
) -> nn.Module:
    weights = _get_torchvision_weights("EfficientNet_V2_S_Weights", pretrained)
    backbone = models.efficientnet_v2_s(weights=weights)

    backbone_out_channels = _infer_backbone_out_channels(backbone.features)

    model = EfficientNetTransformerClassifier(
        backbone=backbone,
        num_classes=num_classes,
        backbone_out_channels=backbone_out_channels,
        grid_size=grid_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        transformer_layers=transformer_layers,
        dropout=dropout,
    )

    return model


def build_model(
    model_name: str,
    num_classes: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    """Build a classification model by name.

    Supported model names
    ---------------------
    - resnet18
    - resnet50
    - densenet121
    - mobilenetv3
    - mobilenet_v3_large
    - efficientnet_b7
    - efficientnetb7
    - efficientnet_v2_s
    - efficientnetv2
    - efficientnet_v2_m
    - efficientnetv2_transformer
    - efficientnet_v2_transformer
    - effnetv2_transformer
    - cnn_transformer
    """
    normalized_name = model_name.lower().replace("-", "_").strip()

    if normalized_name == "resnet18":
        return build_resnet18(num_classes, pretrained, dropout)

    if normalized_name == "resnet50":
        return build_resnet50(num_classes, pretrained, dropout)

    if normalized_name == "densenet121":
        return build_densenet121(num_classes, pretrained, dropout)

    if normalized_name in {"mobilenetv3", "mobilenet_v3", "mobilenet_v3_large"}:
        return build_mobilenet_v3_large(num_classes, pretrained, dropout)

    if normalized_name in {"efficientnet_b7", "efficientnetb7"}:
        return build_efficientnet_b7(num_classes, pretrained, dropout)

    if normalized_name in {"efficientnet_v2_s", "efficientnetv2", "efficientnetv2_s"}:
        return build_efficientnet_v2_s(num_classes, pretrained, dropout)

    if normalized_name in {"efficientnet_v2_m", "efficientnetv2_m"}:
        return build_efficientnet_v2_m(num_classes, pretrained, dropout)

    if normalized_name in {
        "efficientnetv2_transformer",
        "efficientnet_v2_transformer",
        "effnetv2_transformer",
        "cnn_transformer",
        "efficientnet_vit",
        "efficientnetv2_vit",
    }:
        return build_efficientnet_v2_transformer(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
        )

    raise ValueError(
        f"Unsupported model name: {model_name}. "
        "Available models: resnet18, resnet50, densenet121, mobilenetv3, "
        "efficientnet_b7, efficientnet_v2_s, efficientnet_v2_m, "
        "efficientnetv2_transformer."
    )


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove 'module.' prefix from DataParallel checkpoints if present."""
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            cleaned_state_dict[key[len("module.") :]] = value
        else:
            cleaned_state_dict[key] = value

    return cleaned_state_dict


def _extract_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    """Extract a model state dict from different checkpoint formats."""
    if isinstance(checkpoint, dict):
        for key in ["model_state_dict", "state_dict", "model"]:
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        if all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
            return checkpoint

    raise ValueError(
        "Could not extract model weights from checkpoint. Expected one of: "
        "'model_state_dict', 'state_dict', 'model' or a raw state dict."
    )


def load_local_pretrained_weights(
    model: nn.Module,
    weights_path: str | Path | None,
    strict: bool = False,
) -> nn.Module:
    """Load local pretrained weights into a model.

    This helper was added because HPC environments do not always have direct
    internet access. In that case, pretrained weights can be downloaded manually
    and loaded from a local path.

    By default, the function loads only compatible keys. This is useful when the
    checkpoint was trained with a different classification head, because the
    backbone weights can still be reused while ignoring the final classifier.
    """
    if weights_path is None:
        return model

    weights_path = Path(weights_path)

    if not weights_path.exists():
        raise FileNotFoundError(f"Local weights file not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    loaded_state_dict = _strip_module_prefix(_extract_state_dict(checkpoint))
    current_state_dict = model.state_dict()

    compatible_state_dict = {}

    for key, value in loaded_state_dict.items():
        if key in current_state_dict and current_state_dict[key].shape == value.shape:
            compatible_state_dict[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(
        compatible_state_dict,
        strict=False,
    )

    print(
        f"Loaded {len(compatible_state_dict)}/{len(current_state_dict)} "
        f"compatible tensors from local weights: {weights_path}"
    )

    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            "Strict local weight loading failed. "
            f"Missing keys: {missing_keys}. Unexpected keys: {unexpected_keys}."
        )

    return model


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Return total and trainable parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
    }


def freeze_feature_extractor(model: nn.Module) -> nn.Module:
    """Freeze feature extractor layers for transfer learning warm-up."""
    for name, parameter in model.named_parameters():
        if not any(head_name in name for head_name in ["fc", "classifier"]):
            parameter.requires_grad = False

    return model


def unfreeze_all(model: nn.Module) -> nn.Module:
    """Unfreeze all model parameters for fine-tuning."""
    for parameter in model.parameters():
        parameter.requires_grad = True

    return model
