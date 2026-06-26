from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image

from .data import build_transforms
from .models import build_model
from .utils import get_device, load_checkpoint, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on one image")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-path", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    classes = cfg.get("classes") or [str(i) for i in range(int(cfg["num_classes"]))]
    device = get_device()
    model = build_model(cfg["model_name"], num_classes=len(classes), pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    tensor = build_transforms(int(cfg["img_size"]), train=False)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze(0).cpu()
    top_prob, top_idx = probs.max(dim=0)
    print({"predicted_label": classes[int(top_idx)], "confidence": float(top_prob)})


if __name__ == "__main__":
    main()
