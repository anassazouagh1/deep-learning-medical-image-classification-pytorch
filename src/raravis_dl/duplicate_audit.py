from __future__ import annotations

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_images(data_dir: Path):
    for path in data_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def main() -> None:
    parser = argparse.ArgumentParser(description="Find duplicated images using SHA-256 hashes")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="duplicate_report.csv")
    args = parser.parse_args()

    hashes = defaultdict(list)
    for image_path in iter_images(Path(args.data_dir)):
        hashes[file_sha256(image_path)].append(str(image_path))

    duplicates = {h: paths for h, paths in hashes.items() if len(paths) > 1}
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["hash", "num_files", "paths"])
        for h, paths in duplicates.items():
            writer.writerow([h, len(paths), " | ".join(paths)])

    print(f"Images scanned: {sum(len(v) for v in hashes.values())}")
    print(f"Duplicate groups found: {len(duplicates)}")
    print(f"Report saved to: {args.output}")


if __name__ == "__main__":
    main()
