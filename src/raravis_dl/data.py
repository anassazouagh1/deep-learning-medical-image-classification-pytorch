from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, train: bool) -> transforms.Compose:
    """Build image transformations for train/evaluation.

    The augmentations are intentionally conservative because medical images should
    not be distorted aggressively. The same structure can be adapted to
    spectrograms in acoustic classification tasks.
    """
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=8),
            transforms.ColorJitter(brightness=0.08, contrast=0.08),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def compute_class_weights(targets: Iterable[int], num_classes: int) -> torch.Tensor:
    counts = Counter(targets)
    total = sum(counts.values())
    weights = [total / (num_classes * max(counts.get(i, 0), 1)) for i in range(num_classes)]
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(targets: list[int], class_weights: torch.Tensor) -> WeightedRandomSampler:
    sample_weights = [float(class_weights[target]) for target in targets]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


def get_class_distribution(dataset: datasets.ImageFolder) -> dict[str, int]:
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    counts = Counter(dataset.targets)
    return {idx_to_class[idx]: counts.get(idx, 0) for idx in sorted(idx_to_class)}


def build_dataloaders(
    data_dir: str | Path,
    img_size: int,
    batch_size: int,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str], torch.Tensor, dict[str, int]]:
    data_root = Path(data_dir)
    train_ds = datasets.ImageFolder(data_root / "train", transform=build_transforms(img_size, train=True))
    val_ds = datasets.ImageFolder(data_root / "val", transform=build_transforms(img_size, train=False))
    test_ds = datasets.ImageFolder(data_root / "test", transform=build_transforms(img_size, train=False))

    class_weights = compute_class_weights(train_ds.targets, len(train_ds.classes))
    sampler = build_weighted_sampler(train_ds.targets, class_weights) if use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, train_ds.classes, class_weights, get_class_distribution(train_ds)
