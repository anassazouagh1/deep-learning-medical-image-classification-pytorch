from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image

from .data import build_transforms
from .models import build_model
from .utils import ensure_dir, get_device, load_checkpoint, load_config


class ActivationCapture:
    def __init__(self, layer):
        self.activations = None
        self.gradients = None
        self.forward_hook = layer.register_forward_hook(self._forward)
        self.backward_hook = layer.register_full_backward_hook(self._backward)

    def _forward(self, module, inputs, output):
        self.activations = output.detach()

    def _backward(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def close(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def find_last_conv_layer(model: torch.nn.Module):
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv is None:
        raise ValueError("No Conv2d layer found for Grad-CAM")
    return last_conv


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a simple Grad-CAM heatmap")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    device = get_device()

    classes = cfg.get("classes") or [str(i) for i in range(int(cfg["num_classes"]))]
    model = build_model(cfg["model_name"], num_classes=len(classes), pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    transform = build_transforms(int(cfg["img_size"]), train=False)
    tensor = transform(image).unsqueeze(0).to(device)

    target_layer = find_last_conv_layer(model)
    capture = ActivationCapture(target_layer)
    logits = model(tensor)
    predicted_class = int(logits.argmax(dim=1).item())
    score = logits[0, predicted_class]
    model.zero_grad(set_to_none=True)
    score.backward()

    weights = capture.gradients.mean(dim=(2, 3), keepdim=True)
    cam = (weights * capture.activations).sum(dim=1).relu().squeeze().cpu()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    capture.close()

    plt.figure(figsize=(6, 6))
    plt.imshow(image.resize((int(cfg["img_size"]), int(cfg["img_size"]))))
    plt.imshow(cam, alpha=0.45)
    plt.axis("off")
    plt.title(f"Predicted: {classes[predicted_class]}")
    plt.tight_layout()
    plt.savefig(output_dir / "gradcam.png", dpi=180)
    print(f"Saved: {output_dir / 'gradcam.png'}")


if __name__ == "__main__":
    main()
