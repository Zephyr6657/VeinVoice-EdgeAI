from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "segment" / "hand_vein_yolov8n" / "weights" / "best.pt"
DEFAULT_SOURCE = ROOT / "datasets" / "hand_vein" / "test" / "images"
DEFAULT_OUTPUT = ROOT / "datasets" / "hand_vein" / "test" / "pseudo_masks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use a YOLOv8 segmentation checkpoint to batch-generate PNG vein masks."
    )
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to best.pt.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Image file or image directory.")
    parser.add_argument("--out", default=str(DEFAULT_OUTPUT), help="Output directory for PNG masks.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="0", help="Inference device, for example 0 or cpu.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing mask files.",
    )
    return parser.parse_args()


def result_to_mask(result) -> np.ndarray:
    height, width = result.orig_shape
    mask = np.zeros((height, width), dtype=np.uint8)

    if result.masks is None or result.masks.xy is None:
        return mask

    for polygon in result.masks.xy:
        if len(polygon) < 3:
            continue
        points = np.rint(polygon).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)
        cv2.fillPoly(mask, [points], 255)

    return mask


def main() -> None:
    args = parse_args()
    weights = Path(args.weights).resolve()
    source = Path(args.source).resolve()
    out_dir = Path(args.out).resolve()

    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")
    if not source.exists():
        raise FileNotFoundError(f"Prediction source not found: {source}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    written = 0
    skipped = 0

    for result in model.predict(
        source=str(source),
        task="segment",
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        stream=True,
        verbose=False,
    ):
        image_path = Path(result.path)
        mask_path = out_dir / f"{image_path.stem}.png"
        if mask_path.exists() and not args.overwrite:
            skipped += 1
            continue

        mask = result_to_mask(result)
        if not cv2.imwrite(str(mask_path), mask):
            raise ValueError(f"Could not write mask: {mask_path}")
        written += 1

    print(f"Done. Wrote {written} masks to {out_dir}; skipped {skipped} existing masks.")


if __name__ == "__main__":
    main()
