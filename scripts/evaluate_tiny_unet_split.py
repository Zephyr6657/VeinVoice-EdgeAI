from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from train_tiny_unet import TinyUNet


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Tiny U-Net checkpoint on a split dataset.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--dataset", default="datasets/split_hand_vein")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="runs/tiny_unet/split_test_visualizations")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--metrics-out", default=None)
    return parser.parse_args()


def imread_unicode(path: Path, flags: int) -> np.ndarray | None:
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


class SplitDataset(Dataset):
    def __init__(self, root: Path, split: str, imgsz: int) -> None:
        self.images_dir = root / split / "images"
        self.masks_dir = root / split / "masks"
        self.imgsz = imgsz
        self.images = image_paths(self.images_dir)
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image_path = self.images[idx]
        mask_path = self.masks_dir / f"{image_path.stem}.png"
        image = imread_unicode(image_path, cv2.IMREAD_COLOR)
        mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.imgsz, self.imgsz), interpolation=cv2.INTER_NEAREST)
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy((mask > 127).astype(np.float32))[None]
        return image_tensor, mask_tensor, image_path.stem


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = float((pred * target).sum())
    union = float(pred.sum() + target.sum())
    return (2.0 * intersection + eps) / (union + eps)


def colorize_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask] = color
    return canvas


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    color_canvas = colorize_mask(mask, color)
    return np.where(color_canvas > 0, (0.55 * image + 0.45 * color_canvas).astype(np.uint8), image)


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = cv2.copyMakeBorder(image, 30, 0, 0, 0, cv2.BORDER_CONSTANT, value=(18, 18, 18))
    cv2.putText(canvas, title, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1, cv2.LINE_AA)
    return canvas


def imwrite_unicode(path: Path, image: np.ndarray) -> bool:
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        return False
    encoded.tofile(str(path))
    return True


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    dataset_root = Path(args.dataset).resolve()
    out_dir = Path(args.out).resolve()

    ckpt = torch.load(weights_path, map_location="cpu")
    imgsz = int(ckpt["imgsz"])
    base_channels = int(ckpt["base_channels"])
    model = TinyUNet(base_channels=base_channels)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = SplitDataset(dataset_root, args.split, imgsz)
    loader = DataLoader(ds, batch_size=1, shuffle=False)

    out_dir.mkdir(parents=True, exist_ok=True)
    dices = []
    rows = []
    with torch.no_grad():
        for image_tensor, mask_tensor, stem in loader:
            logits = model(image_tensor)
            pred = (torch.sigmoid(logits)[0, 0].numpy() >= args.threshold)
            target = mask_tensor[0, 0].numpy() > 0.5
            d = dice_score(pred, target)
            dices.append(d)
            rows.append((stem[0], d))

            img = (image_tensor[0].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            gt_panel = colorize_mask(target, (0, 220, 255))
            pred_panel = colorize_mask(pred, (255, 80, 80))
            overlay = overlay_mask(img, target, (0, 220, 255))
            overlay = overlay_mask(overlay, pred, (255, 80, 80))
            panels = [
                add_title(img, "image"),
                add_title(gt_panel, "ground truth"),
                add_title(pred_panel, f"prediction dice={d:.3f}"),
                add_title(overlay, "overlay: gt yellow, pred red"),
            ]
            composite = np.concatenate(panels, axis=1)
            out_path = out_dir / f"{stem[0]}_viz.png"
            if not imwrite_unicode(out_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)):
                raise ValueError(f"Could not write {out_path}")

    print(f"mean_dice: {float(np.mean(dices)):.4f}")
    print(f"min_dice: {float(np.min(dices)):.4f}")
    print(f"max_dice: {float(np.max(dices)):.4f}")
    print(f"threshold: {args.threshold:.2f}")
    print(f"visualizations: {out_dir}")
    if args.metrics_out:
        metrics_path = Path(args.metrics_out).resolve()
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        with metrics_path.open("w", encoding="utf-8") as file:
            file.write("stem\tdice\n")
            for stem, dice in sorted(rows, key=lambda item: item[1]):
                file.write(f"{stem}\t{dice:.6f}\n")
        print(f"metrics: {metrics_path}")


if __name__ == "__main__":
    main()
