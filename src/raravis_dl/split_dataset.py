from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a reproducible train/validation/test split for an "
            "ImageFolder-style image classification dataset."
        )
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Raw dataset directory. Expected format: input_dir/class_name/image.png",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory where train/, val/ and test/ folders will be created.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="Training split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to make the split reproducible.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "move"],
        default="copy",
        help="Copy or move images into the output directory.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it already exists.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Optional list of image extensions to include, for example: .jpg .png .jpeg",
    )

    return parser.parse_args()


def normalize_extensions(extensions: list[str] | None) -> set[str]:
    """Normalize image extensions provided by the user."""
    if not extensions:
        return IMAGE_EXTENSIONS

    normalized = set()

    for extension in extensions:
        extension = extension.lower().strip()

        if not extension.startswith("."):
            extension = f".{extension}"

        normalized.add(extension)

    return normalized


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    """Validate that split ratios are correct."""
    total = train_ratio + val_ratio + test_ratio

    if not abs(total - 1.0) < 1e-6:
        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Got train={train_ratio}, val={val_ratio}, "
            f"test={test_ratio}, total={total}"
        )

    for name, value in {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }.items():
        if value <= 0 or value >= 1:
            raise ValueError(f"{name} must be between 0 and 1. Got {value}")


def discover_class_folders(input_dir: str | Path) -> list[Path]:
    """Discover class folders from a raw ImageFolder-style dataset.

    Expected structure:

    input_dir/
        class_1/
            image_1.png
        class_2/
            image_2.png
    """
    root = Path(input_dir)

    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {root}")

    ignored_folders = {"train", "val", "validation", "test"}

    class_folders = [
        path
        for path in root.iterdir()
        if path.is_dir()
        and not path.name.startswith(".")
        and path.name.lower() not in ignored_folders
    ]

    if not class_folders:
        raise ValueError(
            "No class folders were found. "
            "Expected structure: input_dir/class_name/image.png"
        )

    return sorted(class_folders, key=lambda path: path.name.lower())


def collect_images(input_dir: str | Path, extensions: set[str]) -> pd.DataFrame:
    """Collect image paths and class labels from the raw dataset."""
    root = Path(input_dir)
    class_folders = discover_class_folders(root)

    rows: list[dict[str, Any]] = []

    for class_index, class_folder in enumerate(class_folders):
        image_files = [
            path
            for path in class_folder.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        ]

        image_files = sorted(image_files)

        for image_path in image_files:
            rows.append(
                {
                    "original_path": str(image_path),
                    "relative_path": str(image_path.relative_to(root)),
                    "file_name": image_path.name,
                    "class_name": class_folder.name,
                    "class_index": class_index,
                    "extension": image_path.suffix.lower(),
                    "file_size_bytes": int(image_path.stat().st_size),
                }
            )

    if not rows:
        raise ValueError(f"No image files were found inside: {root}")

    return pd.DataFrame(rows)


def can_stratify(labels: list[int], min_required_per_class: int = 3) -> bool:
    """Check if stratification is possible.

    Stratification requires enough samples per class. If one class has too few
    samples, sklearn can fail when creating the split.
    """
    counts = Counter(labels)

    if len(counts) < 2:
        return False

    return all(count >= min_required_per_class for count in counts.values())


def create_split_dataframe(
    files_df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Create a reproducible train/validation/test split.

    The split is stratified by class whenever possible. Stratification helps keep
    class distributions similar across train, validation and test sets.
    """
    files_df = files_df.copy()

    labels = files_df["class_index"].astype(int).tolist()
    stratify_labels = labels if can_stratify(labels) else None

    if stratify_labels is None:
        print(
            "WARNING: Stratified train split is not possible because at least one "
            "class has too few samples. Falling back to random split."
        )

    train_df, temp_df = train_test_split(
        files_df,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_labels,
    )

    relative_test_ratio = test_ratio / (val_ratio + test_ratio)

    temp_labels = temp_df["class_index"].astype(int).tolist()
    temp_stratify = (
        temp_labels if can_stratify(temp_labels, min_required_per_class=2) else None
    )

    if temp_stratify is None:
        print(
            "WARNING: Stratified validation/test split is not possible. "
            "Falling back to random validation/test split."
        )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_ratio,
        random_state=seed,
        shuffle=True,
        stratify=temp_stratify,
    )

    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    val_df["split"] = "val"
    test_df["split"] = "test"

    split_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    split_df = split_df.sort_values(
        by=["split", "class_name", "file_name"]
    ).reset_index(drop=True)

    return split_df


def prepare_output_directory(output_dir: str | Path, overwrite: bool) -> Path:
    """Prepare the output directory where the split dataset will be created."""
    output_dir = Path(output_dir)

    if output_dir.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Use --overwrite to replace it."
            )

        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test"]:
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    return output_dir


def avoid_name_collision(destination: Path) -> Path:
    """Avoid overwriting files when two images share the same file name."""
    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent

    counter = 1
    candidate = destination

    while candidate.exists():
        candidate = parent / f"{stem}_{counter}{suffix}"
        counter += 1

    return candidate


def copy_or_move_file(source: Path, destination: Path, copy_mode: str) -> None:
    """Copy or move one image to the destination split folder."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if copy_mode == "copy":
        shutil.copy2(source, destination)
        return

    if copy_mode == "move":
        shutil.move(str(source), str(destination))
        return

    raise ValueError(f"Unsupported copy mode: {copy_mode}")


def materialize_split(
    split_df: pd.DataFrame,
    output_dir: str | Path,
    copy_mode: str,
) -> pd.DataFrame:
    """Create the train/val/test folders and copy or move the images."""
    output_dir = Path(output_dir)
    split_df = split_df.copy()

    destination_paths: list[str] = []

    for _, row in split_df.iterrows():
        source = Path(row["original_path"])
        split = row["split"]
        class_name = row["class_name"]

        destination = output_dir / split / class_name / source.name

        if destination.exists():
            destination = avoid_name_collision(destination)

        copy_or_move_file(
            source=source,
            destination=destination,
            copy_mode=copy_mode,
        )

        destination_paths.append(str(destination))

    split_df["destination_path"] = destination_paths
    split_df["destination_relative_path"] = [
        str(Path(path).relative_to(output_dir)) for path in destination_paths
    ]

    return split_df


def build_summary(split_df: pd.DataFrame, output_dir: str | Path) -> dict[str, Any]:
    """Build a JSON-friendly summary of the created split."""
    summary: dict[str, Any] = {
        "output_dir": str(output_dir),
        "total_images": int(len(split_df)),
        "classes": sorted(split_df["class_name"].unique().tolist()),
        "class_distribution_total": (
            split_df["class_name"].value_counts().sort_index().astype(int).to_dict()
        ),
        "splits": {},
    }

    for split in ["train", "val", "test"]:
        split_part = split_df[split_df["split"] == split]

        summary["splits"][split] = {
            "num_images": int(len(split_part)),
            "class_distribution": (
                split_part["class_name"]
                .value_counts()
                .sort_index()
                .astype(int)
                .to_dict()
            ),
        }

    return summary


def save_reports(
    split_df: pd.DataFrame,
    summary: dict[str, Any],
    output_dir: str | Path,
) -> None:
    """Save manifest and summary reports for reproducibility."""
    output_dir = Path(output_dir)
    reports_dir = output_dir / "_split_reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    split_df.to_csv(reports_dir / "split_manifest.csv", index=False)

    with (reports_dir / "split_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    summary_rows: list[dict[str, Any]] = []

    for split, split_info in summary["splits"].items():
        for class_name, count in split_info["class_distribution"].items():
            summary_rows.append(
                {
                    "split": split,
                    "class_name": class_name,
                    "count": int(count),
                }
            )

    pd.DataFrame(summary_rows).to_csv(
        reports_dir / "split_class_distribution.csv",
        index=False,
    )


def print_summary(summary: dict[str, Any]) -> None:
    """Print a readable split summary in the terminal."""
    print("Split completed successfully.")
    print("=" * 80)
    print(f"Total images: {summary['total_images']}")

    for split, split_info in summary["splits"].items():
        print(f"{split}: {split_info['num_images']} images")

    print("\nClass distribution by split:")

    for split, split_info in summary["splits"].items():
        print(f"\n{split}")

        for class_name, count in split_info["class_distribution"].items():
            print(f"- {class_name}: {count}")

    print("=" * 80)


def main() -> None:
    args = parse_args()

    validate_ratios(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    extensions = normalize_extensions(args.extensions)

    files_df = collect_images(
        input_dir=args.input_dir,
        extensions=extensions,
    )

    output_dir = prepare_output_directory(
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )

    split_df = create_split_dataframe(
        files_df=files_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    split_df = materialize_split(
        split_df=split_df,
        output_dir=output_dir,
        copy_mode=args.copy_mode,
    )

    summary = build_summary(
        split_df=split_df,
        output_dir=output_dir,
    )

    save_reports(
        split_df=split_df,
        summary=summary,
        output_dir=output_dir,
    )

    print_summary(summary)


if __name__ == "__main__":
    main()
