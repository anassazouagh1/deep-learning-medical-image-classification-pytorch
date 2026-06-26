from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .data import build_dataloaders
from .engine import run_epoch
from .models import build_model, load_local_pretrained_weights
from .utils import ensure_dir, get_device, load_config, save_checkpoint, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN model for medical image classification")
    parser.add_argument("--config", required=True, help="Path to YAML experiment config")
    parser.add_argument("--data-dir", required=True, help="Dataset root with train/val/test folders")
    parser.add_argument("--output-dir", required=True, help="Experiment output folder")
    parser.add_argument("--local-weights", default=None, help="Optional local pretrained weights for offline HPC nodes")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable torchvision pretrained weights")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    save_json(cfg, output_dir / "config_used.json")

    device = get_device()
    train_loader, val_loader, _, classes, class_weights, class_distribution = build_dataloaders(
        args.data_dir,
        img_size=int(cfg["img_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 4)),
        use_weighted_sampler=bool(cfg.get("use_weighted_sampler", False)),
    )

    save_json({"classes": classes, "train_distribution": class_distribution}, output_dir / "dataset_summary.json")

    model = build_model(
        cfg["model_name"],
        num_classes=len(classes),
        pretrained=bool(cfg.get("use_pretrained", True)) and not args.no_pretrained,
    )
    model = load_local_pretrained_weights(model, args.local_weights).to(device)

    if bool(cfg.get("use_class_weights", False)):
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(model.parameters(), lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=int(cfg.get("scheduler_patience", 3)))

    monitor_metric = cfg.get("monitor_metric", "f1_macro")
    best_score = -1.0
    patience = int(cfg.get("early_stopping_patience", 8))
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer, use_amp=bool(cfg.get("use_amp", True)))
        val_metrics = run_epoch(model, val_loader, criterion, device, optimizer=None, use_amp=bool(cfg.get("use_amp", True)))
        scheduler.step(val_metrics[monitor_metric])

        row = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row, indent=2))
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        current_score = float(val_metrics[monitor_metric])
        if current_score > best_score:
            best_score = current_score
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                checkpoint_dir / "best_model.pt",
                classes=classes,
                config=cfg,
                epoch=epoch,
                best_score=best_score,
                monitor_metric=monitor_metric,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}. Best {monitor_metric}: {best_score:.4f}")
            break

    print(f"Training finished. Best validation {monitor_metric}: {best_score:.4f}")


if __name__ == "__main__":
    main()
