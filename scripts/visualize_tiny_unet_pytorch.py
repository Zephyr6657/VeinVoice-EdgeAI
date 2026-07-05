from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from train_tiny_unet import TinyUNet


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize Tiny U-Net PyTorch checkpoint predictions.")
    parser.add_argument("--weights", default="runs/tiny_unet/tiny_unet_128_bc8_finetuned.pth")
    parser.add_argument("--dataset", default="datasets/finetune_hand_vein")
    parser.add_argument("--out", default="runs/tiny_unet/finetune_visualizations")
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


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
    cv2.putText(
        canvas,
        title,
        (8, 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    dataset_root = Path(args.dataset).resolve()
    images_dir = dataset_root / "images"
    masks_dir = dataset_root / "masks"
    out_dir = Path(args.out).resolve()

    checkpoint = torch.load(weights_path, map_location="cpu")
    imgsz = int(checkpoint["imgsz"])
    base_channels = int(checkpoint["base_channels"])
    model = TinyUNet(base_channels=base_channels)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    out_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    with torch.no_grad():
        for image_path in image_paths(images_dir):
            mask_path = masks_dir / f"{image_path.stem}.png"
            bgr = imread_unicode(image_path, cv2.IMREAD_COLOR)
            mask = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE)
            if bgr is None:
                raise ValueError(f"Could not read image: {image_path}")
            if mask is None:
                raise ValueError(f"Could not read mask: {mask_path}")

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
            target = cv2.resize(mask, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST) > 127
            tensor = torch.from_numpy(resized).permute(2, 0, 1)[None].float() / 255.0
            pred = (torch.sigmoid(model(tensor))[0, 0].numpy() > args.threshold)
            dice = dice_score(pred, target)
            scores.append(dice)

            gt_panel = colorize_mask(target, (0, 220, 255))
            pred_panel = colorize_mask(pred, (255, 80, 80))
            overlay = overlay_mask(resized, target, (0, 220, 255))
            overlay = overlay_mask(overlay, pred, (255, 80, 80))
            panels = [
                add_title(resized, "image"),
                add_title(gt_panel, "ground truth"),
                add_title(pred_panel, f"prediction dice={dice:.3f}"),
                add_title(overlay, "overlay: gt yellow, pred red"),
            ]
            composite = np.concatenate(panels, axis=1)
            out_path = out_dir / f"{image_path.stem}_finetuned_viz.png"
            if not imwrite_unicode(out_path, cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)):
                raise ValueError(f"Could not write visualization: {out_path}")

    print(f"Visualizations written to: {out_dir}")
    print(f"images: {len(scores)}")
    print(f"mean_dice: {float(np.mean(scores)):.4f}")
    print(f"min_dice: {float(np.min(scores)):.4f}")
    print(f"max_dice: {float(np.max(scores)):.4f}")


if __name__ == "__main__":
    main()
