from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Labelme polygon JSON files to binary masks.")
    parser.add_argument("--input", default="数据集", help="Folder containing images and Labelme JSON files.")
    parser.add_argument(
        "--out",
        default="datasets/finetune_hand_vein",
        help="Output folder containing images/ and masks/.",
    )
    parser.add_argument("--label", default="vein", help="Label name to convert into foreground mask.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    return parser.parse_args()


def find_image(input_dir: Path, image_path_value: str | None, stem: str) -> Path:
    candidates: list[Path] = []
    if image_path_value:
        candidates.append(input_dir / image_path_value)
        candidates.append(input_dir / Path(image_path_value).name)
    candidates.extend(input_dir / f"{stem}{ext}" for ext in IMAGE_EXTS)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(f"No matching image found for {stem}")


def polygon_points(points: list[list[float]], width: int, height: int) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    return np.rint(polygon).astype(np.int32)


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def convert_one(json_path: Path, input_dir: Path, images_dir: Path, masks_dir: Path, label: str, overwrite: bool) -> None:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_path = find_image(input_dir, data.get("imagePath"), json_path.stem)

    image = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)

    converted_shapes = 0
    for shape in data.get("shapes", []):
        if shape.get("label") != label:
            continue
        if shape.get("shape_type", "polygon") != "polygon":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        cv2.fillPoly(mask, [polygon_points(points, width, height)], 255)
        converted_shapes += 1

    if converted_shapes == 0:
        print(f"Warning: no '{label}' polygons found in {json_path}")

    out_image = images_dir / image_path.name
    out_mask = masks_dir / f"{json_path.stem}.png"
    if not overwrite and (out_image.exists() or out_mask.exists()):
        raise FileExistsError(f"Output exists for {json_path.stem}; use --overwrite to replace.")

    shutil.copy2(image_path, out_image)
    if not imwrite_unicode(out_mask, mask):
        raise ValueError(f"Could not write mask: {out_mask}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    images_dir = out_dir / "images"
    masks_dir = out_dir / "masks"

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No Labelme JSON files found in {input_dir}")

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    for json_path in json_paths:
        convert_one(json_path, input_dir, images_dir, masks_dir, args.label, args.overwrite)

    print(f"Converted {len(json_paths)} Labelme files.")
    print(f"Images: {images_dir}")
    print(f"Masks:  {masks_dir}")


if __name__ == "__main__":
    main()
