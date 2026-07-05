from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    ext = path.suffix or ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def clear_split_dirs(out_dir: Path) -> None:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for split in ("train", "valid", "test"):
        (out_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (out_dir / split / "masks").mkdir(parents=True, exist_ok=True)


def collect_existing_split(root: Path) -> list[tuple[str, Path, Path]]:
    samples = []
    for split in ("train", "valid", "test"):
        images_dir = root / split / "images"
        masks_dir = root / split / "masks"
        if not images_dir.is_dir():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            mask_path = masks_dir / f"{image_path.stem}.png"
            if mask_path.is_file():
                samples.append((f"own_{image_path.stem}", image_path, mask_path))
    return samples


def collect_hand_veins(root: Path) -> list[tuple[str, Path, Path]]:
    hand_root = root / "Hand_Dataset"
    vein_root = root / "Veins_Dataset"
    samples = []
    for image_path in sorted(hand_root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        rel = image_path.relative_to(hand_root)
        mask_path = vein_root / rel
        if mask_path.is_file():
            stem = "_".join(rel.with_suffix("").parts)
            samples.append((f"handveins_{stem}", image_path, mask_path))
    return samples


def write_sample(name: str, image_path: Path, mask_path: Path, out_images: Path, out_masks: Path) -> bool:
    image = imread_unicode(image_path, cv2.IMREAD_COLOR)
    mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
    if image is None or mask is None:
        return False

    mask = ((mask > 127).astype(np.uint8) * 255)
    image_out = out_images / f"{name}.png"
    mask_out = out_masks / f"{name}.png"
    return imwrite_unicode(image_out, image) and imwrite_unicode(mask_out, mask)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a combined train/valid/test dataset.")
    parser.add_argument("--own-split", default="datasets/split_hand_vein")
    parser.add_argument("--hand-veins", default="Hand-Veins-Dataset")
    parser.add_argument("--out", default="datasets/combined_hand_vein")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    own_samples = collect_existing_split(Path(args.own_split))
    hand_samples = collect_hand_veins(Path(args.hand_veins))
    samples = own_samples + hand_samples
    random.Random(args.seed).shuffle(samples)

    out_dir = Path(args.out)
    clear_split_dirs(out_dir)

    train_end = int(len(samples) * args.train_ratio)
    valid_end = train_end + int(len(samples) * args.valid_ratio)
    splits = {
        "train": samples[:train_end],
        "valid": samples[train_end:valid_end],
        "test": samples[valid_end:],
    }

    total_written = 0
    failed = []
    for split, split_samples in splits.items():
        written = 0
        out_images = out_dir / split / "images"
        out_masks = out_dir / split / "masks"
        for name, image_path, mask_path in split_samples:
            if write_sample(name, image_path, mask_path, out_images, out_masks):
                written += 1
            else:
                failed.append((name, str(image_path), str(mask_path)))
        total_written += written
        print(f"{split}: {written}/{len(split_samples)} samples written")

    print(f"Own samples: {len(own_samples)}")
    print(f"Hand-Veins-Dataset samples: {len(hand_samples)}")
    print(f"Total written: {total_written}/{len(samples)}")
    print(f"Failed: {len(failed)}")
    print(f"Output: {out_dir.resolve()}")
    if failed:
        print("First failed samples:")
        for item in failed[:10]:
            print("  " + " | ".join(item))


if __name__ == "__main__":
    main()
