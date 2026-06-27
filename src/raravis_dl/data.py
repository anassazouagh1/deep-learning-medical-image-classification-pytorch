from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def validate_dataset_structure(data_dir: str | Path) -> dict[str, Path]:
    """Validate that the dataset follows an ImageFolder-style structure.

    Expected structure:

    data_dir/
        train/
            class_1/
            class_2/
        val/
            class_1/
            class_2/
        test/
            class_1/
            class_2/

    This validation is useful because many training errors come from wrong
    folder names or incomplete split directories.
    """
    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    split_dirs = {
        "train": root / "train",
        "val": root / "val",
        "test": root / "test",
    }

    for split_name, split_path in split_dirs.items():
        if not split_path.exists():
            raise FileNotFoundError(
                f"Missing '{split_name}' directory inside dataset root: {split_path}"
            )

        class_dirs = [path for path in split_path.iterdir() if path.is_dir()]

        if not class_dirs:
            raise ValueError(
                f"No class folders found inside split '{split_name}': {split_path}"
            )

    return split_dirs


def get_transforms(img_size: int, train: bool = False) -> transforms.Compose:
    """Build image transformations.

    For chest X-rays, the augmentation is intentionally conservative. Strong
    transformations can damage clinically relevant patterns. Horizontal flipping
    is not used by default because laterality can be meaningful in medical
    images.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(degrees=7),
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.03, 0.03),
                    scale=(0.97, 1.03),
                ),
                transforms.ColorJitter(
                    brightness=0.08,
                    contrast=0.08,
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_class_distribution(dataset: datasets.ImageFolder) -> dict[str, int]:
    """Return the number of samples per class."""
    idx_to_class = {idx: class_name for class_name, idx in dataset.class_to_idx.items()}
    targets = [int(target) for _, target in dataset.samples]
    counts = Counter(targets)

    return {
        idx_to_class[class_idx]: int(counts.get(class_idx, 0))
        for class_idx in range(len(idx_to_class))
    }


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    """Compute inverse-frequency class weights.

    These weights can be passed to CrossEntropyLoss to reduce the impact of
    class imbalance. The formula keeps the average scale stable:

    total_samples / (num_classes * class_count)
    """
    targets = torch.tensor([target for _, target in dataset.samples], dtype=torch.long)

    if targets.numel() == 0:
        raise ValueError("Cannot compute class weights from an empty dataset.")

    num_classes = len(dataset.classes)
    class_counts = torch.bincount(targets, minlength=num_classes).float()

    if torch.any(class_counts == 0):
        missing = torch.where(class_counts == 0)[0].tolist()
        raise ValueError(
            f"Some classes have zero samples in the training split: {missing}"
        )

    total_samples = float(targets.numel())
    weights = total_samples / (num_classes * class_counts)

    return weights.float()


def build_weighted_sampler(dataset: datasets.ImageFolder) -> WeightedRandomSampler:
    """Build a WeightedRandomSampler for imbalanced datasets.

    Each image receives a sampling weight based on the inverse frequency of its
    class. This helps the model see minority classes more often during training.
    """
    class_weights = compute_class_weights(dataset)
    sample_weights = [float(class_weights[target]) for _, target in dataset.samples]

    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


def check_class_consistency(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
) -> None:
    """Check that all splits contain the same classes in the same order."""
    train_classes = train_dataset.classes
    val_classes = val_dataset.classes
    test_classes = test_dataset.classes

    if train_classes != val_classes or train_classes != test_classes:
        raise ValueError(
            "Class mismatch between train, val and test splits.\n"
            f"Train classes: {train_classes}\n"
            f"Val classes: {val_classes}\n"
            f"Test classes: {test_classes}"
        )


def dataset_overview(
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    test_dataset: datasets.ImageFolder,
) -> dict[str, Any]:
    """Create a compact dataset summary for logs and experiment reports."""
    return {
        "classes": train_dataset.classes,
        "num_classes": len(train_dataset.classes),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "train_distribution": get_class_distribution(train_dataset),
        "val_distribution": get_class_distribution(val_dataset),
        "test_distribution": get_class_distribution(test_dataset),
    }


def build_dataloaders(
    data_dir: str | Path,
    img_size: int,
    batch_size: int,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
) -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    list[str],
    torch.Tensor,
    dict[str, int],
]:
    """Build train, validation and test DataLoaders.

    Parameters
    ----------
    data_dir:
        Dataset root containing train/, val/ and test/ folders.
    img_size:
        Input image size used by the model.
    batch_size:
        Batch size.
    num_workers:
        Number of workers used by DataLoader.
    use_weighted_sampler:
        Whether to use WeightedRandomSampler for the training split.

    Returns
    -------
    tuple
        train_loader, val_loader, test_loader, classes, class_weights,
        train_class_distribution
    """
    split_dirs = validate_dataset_structure(data_dir)

    train_dataset = datasets.ImageFolder(
        root=split_dirs["train"],
        transform=get_transforms(img_size=img_size, train=True),
    )

    val_dataset = datasets.ImageFolder(
        root=split_dirs["val"],
        transform=get_transforms(img_size=img_size, train=False),
    )

    test_dataset = datasets.ImageFolder(
        root=split_dirs["test"],
        transform=get_transforms(img_size=img_size, train=False),
    )

    check_class_consistency(train_dataset, val_dataset, test_dataset)

    class_weights = compute_class_weights(train_dataset)
    class_distribution = get_class_distribution(train_dataset)

    sampler = build_weighted_sampler(train_dataset) if use_weighted_sampler else None

    shuffle_train = sampler is None

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes,
        class_weights,
        class_distribution,
    )


def print_dataset_summary(data_dir: str | Path, img_size: int = 224) -> None:
    """Small helper to inspect the dataset from a script or notebook."""
    split_dirs = validate_dataset_structure(data_dir)

    train_dataset = datasets.ImageFolder(
        root=split_dirs["train"],
        transform=get_transforms(img_size=img_size, train=False),
    )

    val_dataset = datasets.ImageFolder(
        root=split_dirs["val"],
        transform=get_transforms(img_size=img_size, train=False),
    )

    test_dataset = datasets.ImageFolder(
        root=split_dirs["test"],
        transform=get_transforms(img_size=img_size, train=False),
    )

    check_class_consistency(train_dataset, val_dataset, test_dataset)

    overview = dataset_overview(train_dataset, val_dataset, test_dataset)

    print("Dataset overview")
    print("=" * 80)
    print(f"Classes: {overview['classes']}")
    print(f"Number of classes: {overview['num_classes']}")
    print(f"Train size: {overview['train_size']}")
    print(f"Validation size: {overview['val_size']}")
    print(f"Test size: {overview['test_size']}")

    print("\nTrain distribution")
    for class_name, count in overview["train_distribution"].items():
        print(f"- {class_name}: {count}")

    print("\nValidation distribution")
    for class_name, count in overview["val_distribution"].items():
        print(f"- {class_name}: {count}")

    print("\nTest distribution")
    for class_name, count in overview["test_distribution"].items():
        print(f"- {class_name}: {count}")
