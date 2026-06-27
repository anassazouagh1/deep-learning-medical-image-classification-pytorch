from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from .data import IMAGENET_MEAN, IMAGENET_STD
from .models import build_model
from .utils import ensure_dir, get_device, load_checkpoint, load_config, save_json, set_seed


class GradCAM:
    """Grad-CAM implementation for CNN-based image classifiers.

    Grad-CAM is used to visualize which regions of an input image contributed
    the most to the model prediction. This is especially useful in medical image
    classification because it helps inspect whether the model is focusing on
    meaningful anatomical regions or on irrelevant artifacts.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles: list[Any] = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, int, np.ndarray]:
        """Generate Grad-CAM heatmap.

        Parameters
        ----------
        input_tensor:
            Input image tensor with shape [1, C, H, W].
        target_class:
            Optional class index. If None, the predicted class is used.

        Returns
        -------
        heatmap:
            Normalized Grad-CAM heatmap with values in [0, 1].
        predicted_class:
            Predicted class index.
        probabilities:
            Class probabilities.
        """
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = int(torch.argmax(probabilities, dim=1).item())

        class_idx = predicted_class if target_class is None else int(target_class)
        score = logits[:, class_idx].sum()
        score.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations or gradients.")

        gradients = self.gradients
        activations = self.activations

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

        heatmap = cam.squeeze().detach().cpu().numpy()
        heatmap = normalize_heatmap(heatmap)

        return heatmap, predicted_class, probabilities.squeeze().detach().cpu().numpy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate Grad-CAM and saliency map visualizations for a trained "
            "chest X-ray classification model."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML experiment configuration file.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint file.",
    )
    parser.add_argument(
        "--image-path",
        required=True,
        help="Path to the input image.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where Grad-CAM outputs will be saved.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="Optional target class index. If not provided, the predicted class is used.",
    )
    parser.add_argument(
        "--layer-name",
        default=None,
        help=(
            "Optional target layer name. If not provided, the script automatically "
            "selects the last convolutional block."
        ),
    )
    parser.add_argument(
        "--no-saliency",
        action="store_true",
        help="Disable saliency map generation.",
    )

    return parser.parse_args()


def normalize_heatmap(heatmap: np.ndarray) -> np.ndarray:
    heatmap = heatmap.astype(np.float32)
    heatmap -= np.min(heatmap)
    max_value = np.max(heatmap)

    if max_value > 0:
        heatmap /= max_value

    return heatmap


def load_image(image_path: str | Path, image_size: int) -> tuple[Image.Image, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    tensor = transform(image).unsqueeze(0)
    resized_image = image.resize((image_size, image_size))

    return resized_image, tensor


def image_to_array(image: Image.Image) -> np.ndarray:
    array = np.asarray(image).astype(np.float32) / 255.0
    return np.clip(array, 0.0, 1.0)


def overlay_heatmap(
    image: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.40,
) -> np.ndarray:
    image_array = image_to_array(image)

    cmap = plt.get_cmap("jet")
    colored_heatmap = cmap(heatmap)[..., :3]

    overlay = (1.0 - alpha) * image_array + alpha * colored_heatmap
    return np.clip(overlay, 0.0, 1.0)


def save_image(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    array_uint8 = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
    Image.fromarray(array_uint8).save(path)


def save_heatmap(heatmap: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.imshow(heatmap, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_triptych(
    original: Image.Image,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(original)
    plt.title("Input image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap="jet")
    plt.title("Grad-CAM heatmap")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_saliency_map(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    target_class: int | None = None,
) -> tuple[np.ndarray, int]:
    """Generate a simple gradient-based saliency map."""
    input_tensor = input_tensor.clone().detach().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    predicted_class = int(torch.argmax(logits, dim=1).item())

    class_idx = predicted_class if target_class is None else int(target_class)
    score = logits[:, class_idx].sum()
    score.backward()

    gradients = input_tensor.grad.detach().abs()
    saliency = gradients.max(dim=1)[0].squeeze().cpu().numpy()
    saliency = normalize_heatmap(saliency)

    return saliency, predicted_class


def find_module_by_name(model: torch.nn.Module, layer_name: str) -> torch.nn.Module:
    modules = dict(model.named_modules())

    if layer_name not in modules:
        available = list(modules.keys())
        preview = available[-40:]
        raise ValueError(
            f"Layer '{layer_name}' was not found in the model. "
            f"Last available layers: {preview}"
        )

    return modules[layer_name]


def select_target_layer(
    model: torch.nn.Module,
    model_name: str,
    layer_name: str | None = None,
) -> torch.nn.Module:
    """Select a target layer for Grad-CAM.

    The function supports common torchvision architectures used in the project.
    A manual layer name can also be provided from the command line.
    """
    if layer_name:
        return find_module_by_name(model, layer_name)

    name = model_name.lower().replace("-", "_")

    if "resnet" in name:
        return model.layer4[-1]

    if "densenet" in name:
        return model.features.denseblock4

    if "mobilenet" in name:
        return model.features[-1]

    if "efficientnet" in name:
        return model.features[-1]

    raise ValueError(
        f"Automatic target layer selection is not defined for model '{model_name}'. "
        "Please provide --layer-name manually."
    )


def build_prediction_summary(
    image_path: str | Path,
    classes: list[str],
    predicted_class: int,
    probabilities: np.ndarray,
    target_class: int | None,
    model_name: str,
    image_size: int,
) -> dict[str, Any]:
    sorted_indices = np.argsort(probabilities)[::-1]

    top_predictions = [
        {
            "class_index": int(idx),
            "class_name": classes[int(idx)] if int(idx) < len(classes) else str(idx),
            "probability": float(probabilities[int(idx)]),
        }
        for idx in sorted_indices[: min(5, len(sorted_indices))]
    ]

    return {
        "image_path": str(image_path),
        "model_name": model_name,
        "image_size": image_size,
        "predicted_class_index": int(predicted_class),
        "predicted_class_name": classes[predicted_class]
        if predicted_class < len(classes)
        else str(predicted_class),
        "target_class_index": target_class,
        "target_class_name": classes[target_class]
        if target_class is not None and target_class < len(classes)
        else None,
        "top_predictions": top_predictions,
    }


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    output_dir = ensure_dir(args.output_dir)
    device = get_device()

    model_name = str(cfg["model_name"])
    image_size = int(cfg["img_size"])

    print("=" * 80)
    print("Grad-CAM interpretability")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Image: {args.image_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print("=" * 80)

    checkpoint_metadata = torch.load(args.checkpoint, map_location="cpu")
    classes = checkpoint_metadata.get(
        "classes",
        [
            "Atelectasis",
            "Effusion",
            "Emphysema",
            "No finding",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
        ],
    )

    model = build_model(
        model_name,
        num_classes=len(classes),
        pretrained=False,
        dropout=float(cfg.get("dropout", 0.2)),
    )

    load_checkpoint(model, args.checkpoint, device=torch.device("cpu"))
    model = model.to(device)
    model.eval()

    original_image, input_tensor = load_image(args.image_path, image_size)
    input_tensor = input_tensor.to(device)

    target_layer = select_target_layer(
        model=model,
        model_name=model_name,
        layer_name=args.layer_name,
    )

    gradcam = GradCAM(model=model, target_layer=target_layer)

    try:
        heatmap, predicted_class, probabilities = gradcam(
            input_tensor=input_tensor,
            target_class=args.target_class,
        )
    finally:
        gradcam.remove_hooks()

    overlay = overlay_heatmap(original_image, heatmap)

    image_stem = Path(args.image_path).stem

    save_heatmap(
        heatmap,
        output_dir / f"{image_stem}_gradcam_heatmap.png",
    )

    save_image(
        overlay,
        output_dir / f"{image_stem}_gradcam_overlay.png",
    )

    title = (
        f"Prediction: {classes[predicted_class]} "
        f"({float(probabilities[predicted_class]):.3f})"
    )

    save_triptych(
        original=original_image,
        heatmap=heatmap,
        overlay=overlay,
        path=output_dir / f"{image_stem}_gradcam_summary.png",
        title=title,
    )

    summary = build_prediction_summary(
        image_path=args.image_path,
        classes=classes,
        predicted_class=predicted_class,
        probabilities=probabilities,
        target_class=args.target_class,
        model_name=model_name,
        image_size=image_size,
    )

    if not args.no_saliency:
        saliency, _ = generate_saliency_map(
            model=model,
            input_tensor=input_tensor,
            target_class=args.target_class,
        )

        save_heatmap(
            saliency,
            output_dir / f"{image_stem}_saliency_map.png",
        )

        summary["saliency_map"] = str(output_dir / f"{image_stem}_saliency_map.png")

    summary["gradcam_heatmap"] = str(output_dir / f"{image_stem}_gradcam_heatmap.png")
    summary["gradcam_overlay"] = str(output_dir / f"{image_stem}_gradcam_overlay.png")
    summary["gradcam_summary"] = str(output_dir / f"{image_stem}_gradcam_summary.png")
    summary["checkpoint_metadata"] = {
        key: value
        for key, value in checkpoint_metadata.items()
        if key != "model_state_dict"
    }

    save_json(summary, output_dir / f"{image_stem}_interpretability_summary.json")

    print("Prediction summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nFiles exported")
    print(f"- {output_dir / f'{image_stem}_gradcam_heatmap.png'}")
    print(f"- {output_dir / f'{image_stem}_gradcam_overlay.png'}")
    print(f"- {output_dir / f'{image_stem}_gradcam_summary.png'}")

    if not args.no_saliency:
        print(f"- {output_dir / f'{image_stem}_saliency_map.png'}")

    print(f"- {output_dir / f'{image_stem}_interpretability_summary.json'}")
    print("\nGrad-CAM generation finished successfully.")


if __name__ == "__main__":
    main()
