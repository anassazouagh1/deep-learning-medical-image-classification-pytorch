from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


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
            "Audit an image dataset and detect duplicated files using hash-based "
            "comparison. The script is useful to reduce data leakage risk before "
            "creating train/validation/test splits."
        )
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        help="Root directory containing the image dataset.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="CSV file where duplicate groups will be exported.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional JSON summary output path. If not provided, it is created next to the CSV.",
    )
    parser.add_argument(
        "--check-images",
        action="store_true",
        help="Open images with PIL to detect corrupted or unreadable files.",
    )
    parser.add_argument(
        "--extensions",
        nargs="*",
        default=None,
        help="Optional list of image extensions to include, for example: .jpg .png .jpeg",
    )

    return parser.parse_args()


def normalize_extensions(extensions: list[str] | None) -> set[str]:
    if not extensions:
        return IMAGE_EXTENSIONS

    normalized = set()

    for extension in extensions:
        extension = extension.lower().strip()
        if not extension.startswith("."):
            extension = f".{extension}"
        normalized.add(extension)

    return normalized


def find_image_files(data_dir: str | Path, extensions: set[str]) -> list[Path]:
    root = Path(data_dir)

    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]

    return sorted(files)


def compute_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA256 hash for a file.

    SHA256 is used here to detect exact duplicated files. Exact duplicates are
    especially important to remove before splitting the dataset because they can
    create train/test leakage.
    """
    path = Path(path)
    hasher = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)

    return hasher.hexdigest()


def compute_file_size(path: str | Path) -> int:
    return int(Path(path).stat().st_size)


def validate_image(path: str | Path) -> tuple[bool, str | None]:
    """Check if an image can be opened by PIL."""
    try:
        with Image.open(path) as image:
            image.verify()
        return True, None
    except Exception as exc:
        return False, str(exc)


def infer_split_and_class(path: Path, root: Path) -> tuple[str | None, str | None]:
    """Infer split and class name from a common dataset structure.

    Supported examples:

    dataset/train/Pneumonia/img.png
    dataset/val/Nodule/img.png
    dataset/test/No finding/img.png
    dataset/Pneumonia/img.png
    """
    relative_parts = path.relative_to(root).parts

    known_splits = {"train", "val", "validation", "test"}

    for idx, part in enumerate(relative_parts):
        normalized = part.lower()

        if normalized in known_splits:
            split = "val" if normalized == "validation" else normalized
            class_name = relative_parts[idx + 1] if idx + 1 < len(relative_parts) else None
            return split, class_name

    if len(relative_parts) >= 2:
        return None, relative_parts[-2]

    return None, None


def audit_dataset(
    data_dir: str | Path,
    extensions: set[str],
    check_images: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    root = Path(data_dir)
    image_files = find_image_files(root, extensions)

    if not image_files:
        raise ValueError(f"No image files found inside: {root}")

    file_rows: list[dict[str, Any]] = []
    corrupted_rows: list[dict[str, Any]] = []

    for idx, image_path in enumerate(image_files, start=1):
        split, class_name = infer_split_and_class(image_path, root)

        is_valid = True
        error_message = None

        if check_images:
            is_valid, error_message = validate_image(image_path)

            if not is_valid:
                corrupted_rows.append(
                    {
                        "path": str(image_path),
                        "relative_path": str(image_path.relative_to(root)),
                        "split": split,
                        "class_name": class_name,
                        "error": error_message,
                    }
                )

        sha256 = compute_sha256(image_path)

        file_rows.append(
            {
                "index": idx,
                "path": str(image_path),
                "relative_path": str(image_path.relative_to(root)),
                "file_name": image_path.name,
                "extension": image_path.suffix.lower(),
                "file_size_bytes": compute_file_size(image_path),
                "split": split,
                "class_name": class_name,
                "sha256": sha256,
                "is_valid_image": is_valid,
                "image_error": error_message,
            }
        )

    files_df = pd.DataFrame(file_rows)
    corrupted_df = pd.DataFrame(corrupted_rows)

    duplicate_groups = files_df.groupby("sha256").filter(lambda group: len(group) > 1)

    duplicate_rows: list[dict[str, Any]] = []

    if not duplicate_groups.empty:
        grouped = duplicate_groups.groupby("sha256")

        for group_id, (sha256, group) in enumerate(grouped, start=1):
            group = group.sort_values(by="relative_path")
            reference = group.iloc[0]

            involved_splits = sorted(
                split for split in group["split"].dropna().unique().tolist()
            )

            leakage_risk = len(involved_splits) > 1

            for duplicate_index, (_, row) in enumerate(group.iterrows(), start=1):
                duplicate_rows.append(
                    {
                        "duplicate_group_id": group_id,
                        "duplicate_index": duplicate_index,
                        "sha256": sha256,
                        "relative_path": row["relative_path"],
                        "path": row["path"],
                        "file_name": row["file_name"],
                        "file_size_bytes": row["file_size_bytes"],
                        "split": row["split"],
                        "class_name": row["class_name"],
                        "reference_relative_path": reference["relative_path"],
                        "is_reference": duplicate_index == 1,
                        "num_files_in_group": len(group),
                        "involved_splits": ",".join(involved_splits),
                        "possible_split_leakage": leakage_risk,
                    }
                )

    duplicates_df = pd.DataFrame(duplicate_rows)

    class_distribution = (
        files_df["class_name"]
        .dropna()
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )

    split_distribution = (
        files_df["split"]
        .dropna()
        .value_counts()
        .sort_index()
        .astype(int)
        .to_dict()
    )

    duplicate_file_count = int(len(duplicates_df))
    duplicate_group_count = int(duplicates_df["duplicate_group_id"].nunique()) if not duplicates_df.empty else 0

    duplicated_beyond_reference = 0
    possible_split_leakage_groups = 0

    if not duplicates_df.empty:
        duplicated_beyond_reference = int((duplicates_df["is_reference"] == False).sum())
        possible_split_leakage_groups = int(
            duplicates_df.loc[
                duplicates_df["possible_split_leakage"] == True,
                "duplicate_group_id",
            ].nunique()
        )

    summary = {
        "data_dir": str(root),
        "total_image_files": int(len(files_df)),
        "unique_hashes": int(files_df["sha256"].nunique()),
        "duplicate_groups": duplicate_group_count,
        "duplicate_files_including_references": duplicate_file_count,
        "duplicated_files_beyond_reference": duplicated_beyond_reference,
        "possible_split_leakage_groups": possible_split_leakage_groups,
        "corrupted_or_unreadable_images": int(len(corrupted_df)),
        "extensions_used": sorted(extensions),
        "class_distribution": class_distribution,
        "split_distribution": split_distribution,
    }

    return duplicates_df, corrupted_df, summary


def save_outputs(
    duplicates_df: pd.DataFrame,
    corrupted_df: pd.DataFrame,
    summary: dict[str, Any],
    output_csv: str | Path,
    summary_output: str | Path | None,
) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if duplicates_df.empty:
        duplicates_df = pd.DataFrame(
            columns=[
                "duplicate_group_id",
                "duplicate_index",
                "sha256",
                "relative_path",
                "path",
                "file_name",
                "file_size_bytes",
                "split",
                "class_name",
                "reference_relative_path",
                "is_reference",
                "num_files_in_group",
                "involved_splits",
                "possible_split_leakage",
            ]
        )

    duplicates_df.to_csv(output_csv, index=False)

    corrupted_output = output_csv.with_name(f"{output_csv.stem}_corrupted_images.csv")

    if corrupted_df.empty:
        corrupted_df = pd.DataFrame(
            columns=[
                "path",
                "relative_path",
                "split",
                "class_name",
                "error",
            ]
        )

    corrupted_df.to_csv(corrupted_output, index=False)

    if summary_output is None:
        summary_output = output_csv.with_name(f"{output_csv.stem}_summary.json")

    summary_output = Path(summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)

    with summary_output.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print("Audit finished successfully.")
    print(f"Duplicate report: {output_csv}")
    print(f"Corrupted image report: {corrupted_output}")
    print(f"Summary report: {summary_output}")


def main() -> None:
    args = parse_args()

    extensions = normalize_extensions(args.extensions)

    duplicates_df, corrupted_df, summary = audit_dataset(
        data_dir=args.data_dir,
        extensions=extensions,
        check_images=args.check_images,
    )

    save_outputs(
        duplicates_df=duplicates_df,
        corrupted_df=corrupted_df,
        summary=summary,
        output_csv=args.output,
        summary_output=args.summary_output,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
