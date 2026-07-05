from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = ("train", "valid", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLO polygon segmentation labels to PNG masks."
    )
    parser.add_argument(
        "--dataset",
        default="datasets/hand_vein",
        help="Dataset root containing split/images and split/labels folders.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        help="Dataset splits to convert, for example: train valid test.",
    )
    parser.add_argument(
        "--output-name",
        default="masks",
        help="Output directory name under each split.",
    )
    parser.add_argument(
        "--class-mask",
        action="store_true",
        help="Write class-id masks: background=0, class pixels=class_id+1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files.",
    )
    return parser.parse_args()


def find_images(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def polygon_to_pixels(values: list[str], width: int, height: int) -> np.ndarray:
    coords = np.asarray([float(value) for value in values], dtype=np.float32)
    points = coords.reshape(-1, 2)
    points[:, 0] = np.clip(points[:, 0] * width, 0, width - 1)
    points[:, 1] = np.clip(points[:, 1] * height, 0, height - 1)
    return np.rint(points).astype(np.int32)


def read_label_polygons(label_path: Path, width: int, height: int) -> list[tuple[int, np.ndarray]]:
    polygons: list[tuple[int, np.ndarray]] = []
    if not label_path.is_file():
        return polygons

    for line_no, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            raise ValueError(
                f"{label_path}:{line_no}: expected YOLO polygon row "
                "(class x1 y1 x2 y2 x3 y3 ...)"
            )

        class_id = int(parts[0])
        polygon = polygon_to_pixels(parts[1:], width, height)
        polygons.append((class_id, polygon))

    return polygons


def convert_image(
    image_path: Path,
    labels_dir: Path,
    masks_dir: Path,
    class_mask: bool,
    overwrite: bool,
) -> bool:
    mask_path = masks_dir / f"{image_path.stem}.png"
    if mask_path.exists() and not overwrite:
        return False

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    label_path = labels_dir / f"{image_path.stem}.txt"

    for class_id, polygon in read_label_polygons(label_path, width, height):
        fill_value = class_id + 1 if class_mask else 255
        cv2.fillPoly(mask, [polygon], int(fill_value))

    masks_dir.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(mask_path), mask):
        raise ValueError(f"Could not write mask: {mask_path}")

    return True


def convert_split(
    dataset_root: Path,
    split: str,
    output_name: str,
    class_mask: bool,
    overwrite: bool,
) -> tuple[int, int]:
    images_dir = dataset_root / split / "images"
    labels_dir = dataset_root / split / "labels"
    masks_dir = dataset_root / split / output_name

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing image directory: {images_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing label directory: {labels_dir}")

    image_paths = find_images(images_dir)
    written = 0
    for image_path in image_paths:
        if convert_image(image_path, labels_dir, masks_dir, class_mask, overwrite):
            written += 1

    return len(image_paths), written


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset).resolve()

    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    print(f"Converting YOLO polygons to masks in: {dataset_root}")
    for split in args.splits:
        total, written = convert_split(
            dataset_root=dataset_root,
            split=split,
            output_name=args.output_name,
            class_mask=args.class_mask,
            overwrite=args.overwrite,
        )
        skipped = total - written
        print(f"{split}: {written} masks written, {skipped} skipped, {total} images total")

    mode = "class-id" if args.class_mask else "binary"
    print(f"Done. Wrote {mode} PNG masks under each split's {args.output_name}/ folder.")


if __name__ == "__main__":
    main()
