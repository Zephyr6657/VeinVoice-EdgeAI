from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Labelme files to train/valid/test image-mask splits.")
    parser.add_argument("--input", default="数据集")
    parser.add_argument("--out", default="datasets/split_hand_vein")
    parser.add_argument("--label", default="vein")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


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


def find_image(input_dir: Path, image_path_value: str | None, stem: str) -> Path:
    candidates: list[Path] = []
    candidates.extend(input_dir / f"{stem}{ext}" for ext in IMAGE_EXTS)
    if image_path_value:
        candidates.append(input_dir / image_path_value)
        candidates.append(input_dir / Path(image_path_value).name)

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No matching image found for {stem}")


def polygon_points(points: list[list[float]], width: int, height: int) -> np.ndarray:
    polygon = np.asarray(points, dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
    return np.rint(polygon).astype(np.int32)


def load_item(json_path: Path, input_dir: Path, label: str) -> tuple[Path, np.ndarray, int]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_path = find_image(input_dir, data.get("imagePath"), json_path.stem)
    image = imread_unicode(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    shapes = 0
    for shape in data.get("shapes", []):
        if shape.get("label") != label:
            continue
        if shape.get("shape_type", "polygon") != "polygon":
            continue
        points = shape.get("points", [])
        if len(points) < 3:
            continue
        cv2.fillPoly(mask, [polygon_points(points, width, height)], 255)
        shapes += 1
    return image_path, mask, shapes


def prepare_output(out_dir: Path, overwrite: bool, splits: tuple[str, ...]) -> None:
    if out_dir.exists() and overwrite:
        shutil.rmtree(out_dir)
    for split in splits:
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "masks").mkdir(parents=True, exist_ok=True)


def split_items(items: list[Path], train_ratio: float, valid_ratio: float, seed: int) -> dict[str, list[Path]]:
    items = list(items)
    random.Random(seed).shuffle(items)
    train_count = round(len(items) * train_ratio)
    valid_count = round(len(items) * valid_ratio)
    if valid_count == 0:
        return {
            "train": items[:train_count],
            "test": items[train_count:],
        }
    return {
        "train": items[:train_count],
        "valid": items[train_count : train_count + valid_count],
        "test": items[train_count + valid_count :],
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input).resolve()
    out_dir = Path(args.out).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")
    if args.train_ratio <= 0 or args.valid_ratio < 0 or args.train_ratio + args.valid_ratio >= 1:
        raise ValueError("Ratios must satisfy train > 0, valid >= 0, train + valid < 1")

    json_paths = sorted(input_dir.glob("*.json"))
    if not json_paths:
        raise FileNotFoundError(f"No Labelme JSON files found in {input_dir}")

    splits = split_items(json_paths, args.train_ratio, args.valid_ratio, args.seed)
    prepare_output(out_dir, args.overwrite, tuple(splits.keys()))

    total_empty = 0
    skipped: list[str] = []
    for split, paths in splits.items():
        empty = 0
        written = 0
        for json_path in paths:
            try:
                image_path, mask, shapes = load_item(json_path, input_dir, args.label)
            except (FileNotFoundError, ValueError) as exc:
                skipped.append(f"{json_path.name}: {exc}")
                continue
            if shapes == 0:
                empty += 1
            shutil.copy2(image_path, out_dir / split / "images" / image_path.name)
            mask_path = out_dir / split / "masks" / f"{json_path.stem}.png"
            if not imwrite_unicode(mask_path, mask):
                raise ValueError(f"Could not write mask: {mask_path}")
            written += 1
        total_empty += empty
        print(f"{split}: {written}/{len(paths)} samples written, {empty} empty masks")
    if skipped:
        print("Skipped unreadable/missing samples:")
        for item in skipped:
            print(f"- {item}")
    print(f"Done: {len(json_paths) - len(skipped)}/{len(json_paths)} samples, {total_empty} empty masks total")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
