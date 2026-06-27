from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics


def _move_to_device(
    images: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move a batch to the selected device.

    Keeping this small helper makes the training loop easier to read and also
    centralizes the non_blocking transfer used when CUDA pinned memory is enabled
    in the DataLoader.
    """
    images = images.to(device, non_blocking=True)
    targets = targets.to(device, non_blocking=True)
    return images, targets


def _autocast_enabled(device: torch.device, use_amp: bool) -> bool:
    """Enable automatic mixed precision only when CUDA is available."""
    return bool(use_amp and device.type == "cuda")


def _get_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Convert raw logits into class probabilities."""
    return torch.softmax(logits, dim=1)


def _safe_targets_to_list(targets: Iterable[int] | torch.Tensor) -> list[int]:
    if isinstance(targets, torch.Tensor):
        return targets.detach().cpu().numpy().astype(int).tolist()
    return [int(x) for x in targets]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Run one training or evaluation epoch.

    This function is used by both the training and validation phases.

    When an optimizer is provided, the function performs backpropagation and
    updates the model weights. When optimizer is None, the function runs in
    evaluation mode and only computes predictions and metrics.

    The function returns a dictionary with loss and classification metrics:
    accuracy, precision, recall, macro-F1, weighted-F1, AUC macro OVR and
    specificity.

    Parameters
    ----------
    model:
        PyTorch model.
    loader:
        DataLoader for train, validation or test split.
    criterion:
        Loss function, usually CrossEntropyLoss.
    device:
        CPU or CUDA device.
    optimizer:
        Optimizer used during training. If None, the epoch is run as evaluation.
    use_amp:
        Enables automatic mixed precision when running on CUDA.

    Returns
    -------
    dict
        Dictionary containing average loss and classification metrics.
    """
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []

    amp_enabled = _autocast_enabled(device, use_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    progress = tqdm(
        loader,
        desc="train" if is_train else "eval",
        leave=False,
        dynamic_ncols=True,
    )

    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for batch_idx, batch in enumerate(progress):
            images, targets = batch
            images, targets = _move_to_device(images, targets, device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)

            batch_size = targets.size(0)

            if is_train:
                scaler.scale(loss).backward()

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                scaler.step(optimizer)
                scaler.update()

            probabilities = _get_probabilities(logits)
            predictions = torch.argmax(probabilities, dim=1)

            total_loss += float(loss.detach().cpu().item()) * batch_size
            total_samples += batch_size

            all_targets.extend(_safe_targets_to_list(targets))
            all_predictions.extend(_safe_targets_to_list(predictions))
            all_probabilities.extend(
                probabilities.detach().cpu().numpy().astype(float).tolist()
            )

            running_loss = total_loss / max(total_samples, 1)
            running_accuracy = float(
                np.mean(np.asarray(all_targets) == np.asarray(all_predictions))
            )

            progress.set_postfix(
                {
                    "loss": f"{running_loss:.4f}",
                    "acc": f"{running_accuracy:.4f}",
                }
            )

    average_loss = total_loss / max(total_samples, 1)

    num_classes = None
    if all_probabilities:
        num_classes = len(all_probabilities[0])

    metrics = compute_metrics(
        y_true=all_targets,
        y_pred=all_predictions,
        y_prob=all_probabilities if all_probabilities else None,
        num_classes=num_classes,
    )

    metrics = {
        "loss": float(average_loss),
        **metrics,
        "num_samples": int(total_samples),
    }

    return metrics


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> dict[str, Any]:
    """Generate predictions for a full DataLoader.

    This helper is useful for evaluation scripts because it returns the raw
    information needed to create prediction CSV files, confusion matrices and
    later error analysis.

    Returns
    -------
    dict
        Dictionary with targets, predictions and probabilities.
    """
    model.eval()

    all_targets: list[int] = []
    all_predictions: list[int] = []
    all_probabilities: list[list[float]] = []

    amp_enabled = _autocast_enabled(device, use_amp)

    with torch.no_grad():
        for images, targets in tqdm(loader, desc="predict", leave=False, dynamic_ncols=True):
            images, targets = _move_to_device(images, targets, device)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                probabilities = _get_probabilities(logits)
                predictions = torch.argmax(probabilities, dim=1)

            all_targets.extend(_safe_targets_to_list(targets))
            all_predictions.extend(_safe_targets_to_list(predictions))
            all_probabilities.extend(
                probabilities.detach().cpu().numpy().astype(float).tolist()
            )

    return {
        "targets": all_targets,
        "predictions": all_predictions,
        "probabilities": all_probabilities,
    }


def evaluate_with_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a model and return both metrics and raw predictions.

    This function is useful when the evaluation script needs metrics and also
    needs to export a prediction-level CSV file.

    Returns
    -------
    tuple
        metrics, predictions
    """
    metrics = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        use_amp=use_amp,
    )

    predictions = predict_loader(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
    )

    return metrics, predictions
