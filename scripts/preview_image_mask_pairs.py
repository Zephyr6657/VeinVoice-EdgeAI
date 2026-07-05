from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a contact sheet preview for image/mask pairs.")
    parser.add_argument("--dataset", default="datasets/finetune_hand_vein")
    parser.add_argument("--out", default="runs/finetune_hand_vein_mask_preview.png")
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--thumb", type=int, default=192)
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


def image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(image)
    color[mask > 127] = (0, 0, 255)
    return np.where(color > 0, (0.55 * image + 0.45 * color).astype(np.uint8), image)


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset)
    images_dir = dataset / "images"
    masks_dir = dataset / "masks"
    out_path = Path(args.out)

    rows = []
    for image_path in image_paths(images_dir)[: args.limit]:
        mask_path = masks_dir / f"{image_path.stem}.png"
        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        image = cv2.resize(image, (args.thumb, args.thumb), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (args.thumb, args.thumb), interpolation=cv2.INTER_NEAREST)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        rows.append(np.concatenate([image, mask_bgr, overlay(image, mask)], axis=1))

    if not rows:
        raise FileNotFoundError(f"No image/mask pairs found in {dataset}")

    sheet = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not imwrite_unicode(out_path, sheet):
        raise ValueError(f"Could not write preview: {out_path}")
    print(f"Preview written to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
