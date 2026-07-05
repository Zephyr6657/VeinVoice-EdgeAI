from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a YOLOv8 segmentation dataset for hand vein training."
    )
    parser.add_argument(
        "--dataset",
        default="datasets/hand_vein",
        help="Dataset root containing train/valid/test and data.yaml.",
    )
    parser.add_argument(
        "--nc",
        type=int,
        default=1,
        help="Number of classes. Class ids must be in [0, nc - 1].",
    )
    return parser.parse_args()


def label_errors(label_path: Path, nc: int) -> list[str]:
    errors: list[str] = []
    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) == 5:
            errors.append(
                f"line {line_no}: looks like detection box format "
                "(class x_center y_center width height), not polygon segmentation"
            )
            continue

        if len(parts) < 7:
            errors.append(
                f"line {line_no}: expected class id plus at least 3 polygon points, "
                f"got {len(parts)} values"
            )
            continue

        if (len(parts) - 1) % 2 != 0:
            errors.append(
                f"line {line_no}: polygon coordinate count must be even, "
                f"got {len(parts) - 1}"
            )
            continue

        try:
            cls = int(parts[0])
        except ValueError:
            errors.append(f"line {line_no}: class id is not an integer: {parts[0]}")
            continue

        if cls < 0 or cls >= nc:
            errors.append(f"line {line_no}: class id {cls} is outside [0, {nc - 1}]")

        try:
            coords = [float(v) for v in parts[1:]]
        except ValueError as exc:
            errors.append(f"line {line_no}: non-numeric polygon coordinate: {exc}")
            continue

        for index, value in enumerate(coords, start=1):
            if value < 0.0 or value > 1.0:
                errors.append(
                    f"line {line_no}: coordinate #{index}={value} is outside [0, 1]"
                )

    return errors


def check_split(dataset_root: Path, split: str, nc: int) -> tuple[int, int, list[str]]:
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"
    errors: list[str] = []

    if not images_dir.is_dir():
        errors.append(f"{split}: missing image directory: {images_dir}")
        return 0, 0, errors
    if not labels_dir.is_dir():
        errors.append(f"{split}: missing label directory: {labels_dir}")
        return 0, 0, errors

    images_by_stem: dict[str, list[Path]] = defaultdict(list)
    for image_path in images_dir.iterdir():
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTS:
            images_by_stem[image_path.stem].append(image_path)

    labels_by_stem: dict[str, Path] = {
        label_path.stem: label_path
        for label_path in labels_dir.iterdir()
        if label_path.is_file() and label_path.suffix.lower() == ".txt"
    }

    for stem, image_paths in sorted(images_by_stem.items()):
        if len(image_paths) > 1:
            paths = ", ".join(path.name for path in image_paths)
            errors.append(f"{split}: duplicate image stem '{stem}': {paths}")
        if stem not in labels_by_stem:
            errors.append(f"{split}: missing label for image: {image_paths[0].name}")

    for stem, label_path in sorted(labels_by_stem.items()):
        if stem not in images_by_stem:
            errors.append(f"{split}: label has no matching image: {label_path.name}")
        for error in label_errors(label_path, nc):
            errors.append(f"{split}/{label_path.name}: {error}")

    return len(images_by_stem), len(labels_by_stem), errors


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()
    all_errors: list[str] = []

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.is_file():
        all_errors.append(f"missing data.yaml: {data_yaml}")

    print(f"Checking dataset: {dataset_root}")
    for split in SPLITS:
        image_count, label_count, errors = check_split(dataset_root, split, args.nc)
        print(f"{split}: {image_count} images, {label_count} labels")
        all_errors.extend(errors)

    if all_errors:
        print("\nErrors:")
        for error in all_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nDataset check passed. Labels use YOLOv8 polygon segmentation format.")


if __name__ == "__main__":
    main()
