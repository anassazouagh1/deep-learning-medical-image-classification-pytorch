from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from .data import build_dataloaders
from .engine import run_epoch
from .metrics import compute_metrics, per_class_report
from .models import build_model
from .utils import ensure_dir, get_device, load_checkpoint, load_config, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def collect_predictions(model, loader, device):
    model.eval()
    y_true, y_pred, y_prob, paths = [], [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
    # ImageFolder keeps sample paths in the same order when shuffle=False
    paths = [sample[0] for sample in loader.dataset.samples]
    return y_true, y_pred, y_prob, paths


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = ensure_dir(args.output_dir)
    device = get_device()

    _, _, test_loader, classes, _, _ = build_dataloaders(
        args.data_dir,
        img_size=int(cfg["img_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 4)),
        use_weighted_sampler=False,
    )

    model = build_model(cfg["model_name"], num_classes=len(classes), pretrained=False).to(device)
    load_checkpoint(model, args.checkpoint, device)

    y_true, y_pred, y_prob, paths = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob, num_classes=len(classes))
    save_json(metrics, output_dir / "metrics_test.json")
    save_json({"per_class": per_class_report(y_true, y_pred, classes)}, output_dir / "per_class_report.json")

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(output_dir / "confusion_matrix_test.csv")

    rows = []
    for path, true_idx, pred_idx, probs in zip(paths, y_true, y_pred, y_prob):
        row = {
            "path": path,
            "true_label": classes[true_idx],
            "predicted_label": classes[pred_idx],
            "confidence": max(probs),
        }
        row.update({f"prob_{cls}": prob for cls, prob in zip(classes, probs)})
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "predictions_test.csv", index=False)

    print(metrics)


if __name__ == "__main__":
    main()
