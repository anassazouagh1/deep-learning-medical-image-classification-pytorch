from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau

from .data import build_dataloaders
from .engine import run_epoch
from .models import build_model, load_local_pretrained_weights
from .utils import (
    ensure_dir,
    get_device,
    load_config,
    load_checkpoint,
    save_checkpoint,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a Deep Learning model for multi-class chest X-ray classification. "
            "The script supports transfer learning, class weighting, weighted sampling, "
            "early stopping, checkpointing and reproducible experiment logging."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML experiment configuration file.",
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Dataset root directory containing train/, val/ and test/ folders.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where checkpoints, logs and metrics will be saved.",
    )
    parser.add_argument(
        "--local-weights",
        default=None,
        help=(
            "Optional local pretrained weights file. Useful for HPC environments "
            "without direct internet access."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Optional checkpoint path to resume model weights from.",
    )
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable torchvision pretrained weights.",
    )
    parser.add_argument(
        "--evaluate-test-at-end",
        action="store_true",
        help="Run a final test evaluation using the best checkpoint after training.",
    )

    return parser.parse_args()


def to_serializable(value: Any) -> Any:
    """Convert values to JSON-friendly Python types."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    if isinstance(value, dict):
        return {k: to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_serializable(v) for v in value]
    return value


def save_training_environment(output_dir: Path, device: torch.device) -> None:
    """Save basic environment information to make experiments easier to reproduce."""
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    save_json(env, output_dir / "environment.json")


def build_optimizer(model: torch.nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = str(cfg.get("optimizer", "adamw")).lower()
    learning_rate = float(cfg["learning_rate"])
    weight_decay = float(cfg.get("weight_decay", 0.0))

    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    if optimizer_name == "sgd":
        return SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
        )

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict[str, Any],
    epochs: int,
):
    scheduler_name = str(cfg.get("scheduler", "reduce_on_plateau")).lower()

    if scheduler_name in {"reduce_on_plateau", "plateau"}:
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(cfg.get("scheduler_factor", 0.5)),
            patience=int(cfg.get("scheduler_patience", 3)),
        )

    if scheduler_name in {"cosine", "cosine_annealing"}:
        return CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(cfg.get("min_learning_rate", 1e-6)),
        )

    if scheduler_name in {"none", "disabled", "false"}:
        return None

    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def step_scheduler(
    scheduler,
    scheduler_name: str,
    monitor_value: float,
) -> None:
    if scheduler is None:
        return

    if scheduler_name in {"reduce_on_plateau", "plateau"}:
        scheduler.step(monitor_value)
    else:
        scheduler.step()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    output_dir = ensure_dir(args.output_dir)
    checkpoint_dir = ensure_dir(output_dir / "checkpoints")
    logs_dir = ensure_dir(output_dir / "logs")

    save_json(cfg, output_dir / "config_used.json")

    device = get_device()
    save_training_environment(output_dir, device)

    print("=" * 80)
    print("Chest X-Ray Multi-Class Classification Training")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Seed: {seed}")
    print("=" * 80)

    train_loader, val_loader, test_loader, classes, class_weights, class_distribution = build_dataloaders(
        args.data_dir,
        img_size=int(cfg["img_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 4)),
        use_weighted_sampler=bool(cfg.get("use_weighted_sampler", False)),
    )

    dataset_summary = {
        "classes": classes,
        "num_classes": len(classes),
        "train_size": len(train_loader.dataset),
        "val_size": len(val_loader.dataset),
        "test_size": len(test_loader.dataset),
        "train_distribution": class_distribution,
        "class_weights": class_weights.tolist(),
        "use_weighted_sampler": bool(cfg.get("use_weighted_sampler", False)),
        "use_class_weights": bool(cfg.get("use_class_weights", False)),
    }
    save_json(dataset_summary, output_dir / "dataset_summary.json")

    print("Dataset summary")
    print(json.dumps(dataset_summary, indent=2))

    model = build_model(
        cfg["model_name"],
        num_classes=len(classes),
        pretrained=bool(cfg.get("use_pretrained", True)) and not args.no_pretrained,
        dropout=float(cfg.get("dropout", 0.2)),
    )

    model = load_local_pretrained_weights(model, args.local_weights)

    if args.resume:
        print(f"Loading checkpoint to resume weights from: {args.resume}")
        load_checkpoint(model, args.resume, device=torch.device("cpu"))

    model = model.to(device)

    if bool(cfg.get("use_class_weights", False)):
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        print("Using class-weighted CrossEntropyLoss.")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard CrossEntropyLoss.")

    optimizer = build_optimizer(model, cfg)

    epochs = int(cfg["epochs"])
    scheduler_name = str(cfg.get("scheduler", "reduce_on_plateau")).lower()
    scheduler = build_scheduler(optimizer, cfg, epochs)

    monitor_metric = str(cfg.get("monitor_metric", "f1_macro"))
    early_stopping_patience = int(cfg.get("early_stopping_patience", 8))
    use_amp = bool(cfg.get("use_amp", True))

    best_score = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        print("\n" + "-" * 80)
        print(f"Epoch {epoch}/{epochs}")
        print("-" * 80)

        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            use_amp=use_amp,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            use_amp=use_amp,
        )

        if monitor_metric not in val_metrics:
            raise KeyError(
                f"Monitor metric '{monitor_metric}' was not found in validation metrics. "
                f"Available metrics: {list(val_metrics.keys())}"
            )

        current_score = float(val_metrics[monitor_metric])
        step_scheduler(scheduler, scheduler_name, current_score)

        current_lr = float(optimizer.param_groups[0]["lr"])
        epoch_time = time.time() - epoch_start

        row = {
            "epoch": epoch,
            "learning_rate": current_lr,
            "epoch_time_seconds": round(epoch_time, 2),
            **{f"train_{k}": to_serializable(v) for k, v in train_metrics.items()},
            **{f"val_{k}": to_serializable(v) for k, v in val_metrics.items()},
        }

        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        with open(logs_dir / "training_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(to_serializable(row)) + "\n")

        print(json.dumps(to_serializable(row), indent=2))

        improved = current_score > best_score

        if improved:
            best_score = current_score
            best_epoch = epoch
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                checkpoint_dir / "best_model.pt",
                classes=classes,
                config=cfg,
                epoch=epoch,
                best_score=best_score,
                monitor_metric=monitor_metric,
                class_to_idx=getattr(train_loader.dataset, "class_to_idx", None),
            )

            print(
                f"New best model saved at epoch {epoch} "
                f"with validation {monitor_metric} = {best_score:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No improvement for {epochs_without_improvement}/"
                f"{early_stopping_patience} epochs."
            )

        save_checkpoint(
            model,
            checkpoint_dir / "last_model.pt",
            classes=classes,
            config=cfg,
            epoch=epoch,
            best_score=best_score,
            monitor_metric=monitor_metric,
            class_to_idx=getattr(train_loader.dataset, "class_to_idx", None),
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best epoch: {best_epoch}. "
                f"Best validation {monitor_metric}: {best_score:.4f}"
            )
            break

    total_time = time.time() - start_time

    summary = {
        "experiment_name": cfg.get("experiment_name"),
        "model_name": cfg.get("model_name"),
        "image_size": cfg.get("img_size"),
        "batch_size": cfg.get("batch_size"),
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_metric": monitor_metric,
        "best_validation_score": best_score,
        "total_training_time_seconds": round(total_time, 2),
        "total_training_time_minutes": round(total_time / 60, 2),
        "device": str(device),
        "classes": classes,
    }

    save_json(to_serializable(summary), output_dir / "training_summary.json")

    print("\n" + "=" * 80)
    print("Training finished")
    print("=" * 80)
    print(json.dumps(to_serializable(summary), indent=2))

    if args.evaluate_test_at_end:
        print("\nRunning final test evaluation with best checkpoint...")
        best_checkpoint = checkpoint_dir / "best_model.pt"

        if not best_checkpoint.exists():
            raise FileNotFoundError(f"Best checkpoint not found: {best_checkpoint}")

        load_checkpoint(model, best_checkpoint, device)
        test_metrics = run_epoch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            use_amp=use_amp,
        )

        save_json(to_serializable(test_metrics), output_dir / "test_metrics_from_training.json")
        print("Final test metrics:")
        print(json.dumps(to_serializable(test_metrics), indent=2))


if __name__ == "__main__":
    main()
