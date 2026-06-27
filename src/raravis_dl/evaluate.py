from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix

from .data import build_dataloaders
from .engine import evaluate_with_loss
from .metrics import per_class_report
from .models import build_model
from .utils import ensure_dir, get_device, load_checkpoint, load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Deep Learning model for chest X-ray classification. "
            "The script exports global metrics, per-class metrics, predictions and "
            "confusion matrices for later analysis."
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
        "--checkpoint",
        required=True,
        help="Path to trained checkpoint file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where evaluation outputs will be saved.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable automatic mixed precision during evaluation.",
    )

    return parser.parse_args()


def _to_serializable(value: Any) -> Any:
    """Convert NumPy and PyTorch values into JSON-friendly Python objects."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _to_serializable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_serializable(v) for v in value]
    return value


def _select_loader(
    split: str,
    train_loader,
    val_loader,
    test_loader,
):
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    if split == "test":
        return test_loader
    raise ValueError(f"Unsupported split: {split}")


def _build_predictions_dataframe(
    predictions: dict[str, Any],
    classes: list[str],
) -> pd.DataFrame:
    """Create a prediction-level dataframe.

    The output is useful for error analysis because every row contains the real
    label, the predicted label and the probability assigned to each class.
    """
    targets = predictions["targets"]
    preds = predictions["predictions"]
    probs = predictions["probabilities"]

    rows: list[dict[str, Any]] = []

    for idx, (target, pred, prob_vector) in enumerate(zip(targets, preds, probs)):
        row: dict[str, Any] = {
            "sample_index": idx,
            "true_index": int(target),
            "true_label": classes[int(target)],
            "predicted_index": int(pred),
            "predicted_label": classes[int(pred)],
            "is_correct": int(target) == int(pred),
            "confidence": float(max(prob_vector)),
        }

        for class_idx, class_name in enumerate(classes):
            safe_name = (
                class_name.lower()
                .replace(" ", "_")
                .replace("/", "_")
                .replace("-", "_")
            )
            row[f"prob_{safe_name}"] = float(prob_vector[class_idx])

        rows.append(row)

    return pd.DataFrame(rows)


def _save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    classes: list[str],
    output_path: Path,
) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))

    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{name}" for name in classes],
        columns=[f"pred_{name}" for name in classes],
    )

    cm_df.to_csv(output_path)
    return cm_df


def _save_error_analysis(
    predictions_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save only incorrect predictions ordered by confidence.

    This is useful to inspect cases where the model was confidently wrong.
    """
    errors = predictions_df[predictions_df["is_correct"] == False].copy()
    errors = errors.sort_values(by="confidence", ascending=False)
    errors.to_csv(output_path, index=False)


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))

    output_dir = ensure_dir(args.output_dir)

    device = get_device()
    use_amp = bool(cfg.get("use_amp", True)) and not args.no_amp

    print("=" * 80)
    print("Chest X-Ray Multi-Class Classification Evaluation")
    print("=" * 80)
    print(f"Config: {args.config}")
    print(f"Data directory: {args.data_dir}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Split: {args.split}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"AMP enabled: {use_amp}")
    print("=" * 80)

    train_loader, val_loader, test_loader, classes, class_weights, class_distribution = build_dataloaders(
        args.data_dir,
        img_size=int(cfg["img_size"]),
        batch_size=int(cfg["batch_size"]),
        num_workers=int(cfg.get("num_workers", 4)),
        use_weighted_sampler=False,
    )

    loader = _select_loader(
        args.split,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
    )

    model = build_model(
        cfg["model_name"],
        num_classes=len(classes),
        pretrained=False,
        dropout=float(cfg.get("dropout", 0.2)),
    )

    checkpoint_metadata = load_checkpoint(model, args.checkpoint, device=torch.device("cpu"))
    model = model.to(device)
    model.eval()

    if bool(cfg.get("use_class_weights", False)):
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = nn.CrossEntropyLoss()

    metrics, predictions = evaluate_with_loss(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        use_amp=use_amp,
    )

    predictions_df = _build_predictions_dataframe(predictions, classes)

    metrics_output = {
        "split": args.split,
        "model_name": cfg.get("model_name"),
        "experiment_name": cfg.get("experiment_name"),
        "image_size": cfg.get("img_size"),
        "batch_size": cfg.get("batch_size"),
        "num_classes": len(classes),
        "classes": classes,
        "num_samples": len(predictions_df),
        "class_distribution_train": class_distribution,
        "checkpoint": str(args.checkpoint),
        "checkpoint_metadata": {
            key: _to_serializable(value)
            for key, value in checkpoint_metadata.items()
            if key != "model_state_dict"
        },
        "metrics": _to_serializable(metrics),
    }

    save_json(metrics_output, output_dir / f"metrics_{args.split}.json")

    predictions_df.to_csv(output_dir / f"predictions_{args.split}.csv", index=False)

    cm_df = _save_confusion_matrix(
        y_true=predictions["targets"],
        y_pred=predictions["predictions"],
        classes=classes,
        output_path=output_dir / f"confusion_matrix_{args.split}.csv",
    )

    report = per_class_report(
        y_true=predictions["targets"],
        y_pred=predictions["predictions"],
        classes=classes,
    )

    pd.DataFrame(report).to_csv(output_dir / f"per_class_report_{args.split}.csv", index=False)

    _save_error_analysis(
        predictions_df=predictions_df,
        output_path=output_dir / f"errors_{args.split}.csv",
    )

    summary_rows = [
        {
            "metric": key,
            "value": value,
        }
        for key, value in metrics.items()
    ]

    pd.DataFrame(summary_rows).to_csv(output_dir / f"metrics_{args.split}.csv", index=False)

    print("\nEvaluation metrics")
    print(json.dumps(_to_serializable(metrics), indent=2))

    print("\nConfusion matrix")
    print(cm_df)

    print("\nFiles exported")
    print(f"- {output_dir / f'metrics_{args.split}.json'}")
    print(f"- {output_dir / f'metrics_{args.split}.csv'}")
    print(f"- {output_dir / f'predictions_{args.split}.csv'}")
    print(f"- {output_dir / f'confusion_matrix_{args.split}.csv'}")
    print(f"- {output_dir / f'per_class_report_{args.split}.csv'}")
    print(f"- {output_dir / f'errors_{args.split}.csv'}")

    print("\nEvaluation finished successfully.")


if __name__ == "__main__":
    main()
