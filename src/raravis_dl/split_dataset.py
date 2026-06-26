from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def copy_files(files, output_root: Path, split: str, class_name: str) -> None:
    target_dir = output_root / split / class_name
    target_dir.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, target_dir / file.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test split from class folders")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if round(args.train_ratio + args.val_ratio + args.test_ratio, 6) != 1.0:
        raise ValueError("Split ratios must sum to 1.0")

    random.seed(args.seed)
    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    for class_dir in sorted([p for p in input_root.iterdir() if p.is_dir()]):
        files = [p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
        random.shuffle(files)
        n = len(files)
        n_train = int(n * args.train_ratio)
        n_val = int(n * args.val_ratio)
        copy_files(files[:n_train], output_root, "train", class_dir.name)
        copy_files(files[n_train:n_train + n_val], output_root, "val", class_dir.name)
        copy_files(files[n_train + n_val:], output_root, "test", class_dir.name)
        print(class_dir.name, {"train": n_train, "val": n_val, "test": n - n_train - n_val})


if __name__ == "__main__":
    main()
