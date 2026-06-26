from __future__ import annotations

from typing import Any

import torch
from torch import nn
from tqdm import tqdm

from .metrics import compute_metrics


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    use_amp: bool = True,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    total_loss = 0.0
    y_true, y_pred, y_prob = [], [], []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(is_train):
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        probs = torch.softmax(logits.detach(), dim=1)
        preds = torch.argmax(probs, dim=1)
        total_loss += float(loss.item()) * images.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().numpy().tolist())

    metrics = compute_metrics(y_true, y_pred, y_prob=y_prob, num_classes=len(loader.dataset.classes))
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics
