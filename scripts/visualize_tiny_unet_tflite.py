from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs" / "tiny_unet" / "tiny_unet_128_bc8_int8.tflite"
DEFAULT_OUT = ROOT / "runs" / "tiny_unet" / "test_visualizations"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize int8 Tiny U-Net TFLite predictions.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dataset", default="datasets/hand_vein")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def dice_score(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7) -> float:
    pred = pred.astype(np.float32)
    target = target.astype(np.float32)
    intersection = float((pred * target).sum())
    union = float(pred.sum() + target.sum())
    return (2.0 * intersection + eps) / (union + eps)


def quantize_input(image: np.ndarray, input_detail: dict) -> np.ndarray:
    scale, zero_point = input_detail["quantization"]
    if scale == 0:
        raise ValueError("Input quantization scale is zero.")
    quantized = np.rint(image / scale + zero_point)
    dtype = input_detail["dtype"]
    info = np.iinfo(dtype)
    return np.clip(quantized, info.min, info.max).astype(dtype)


def dequantize_output(output: np.ndarray, output_detail: dict) -> np.ndarray:
    scale, zero_point = output_detail["quantization"]
    return (output.astype(np.float32) - zero_point) * scale


def image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def colorize_mask(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    canvas = np.zeros((*mask.shape, 3), dtype=np.uint8)
    canvas[mask] = color
    return canvas


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    overlay = image.copy()
    color_canvas = colorize_mask(mask, color)
    return np.where(color_canvas > 0, (0.55 * image + 0.45 * color_canvas).astype(np.uint8), overlay)


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
    model_path = Path(args.model).resolve()
    dataset_root = Path(args.dataset).resolve()
    images_dir = dataset_root / args.split / "images"
    masks_dir = dataset_root / args.split / "masks"
    out_dir = Path(args.out).resolve()

    if not model_path.is_file():
        raise FileNotFoundError(f"TFLite model not found: {model_path}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not masks_dir.is_dir():
        raise FileNotFoundError(f"Mask directory not found: {masks_dir}")

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    _, height, width, channels = input_detail["shape"]
    if channels != 3:
        raise ValueError(f"Expected 3-channel input, got shape {input_detail['shape']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for image_path in image_paths(images_dir)[: args.limit]:
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        gt = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if bgr is None:
            raise ValueError(f"Could not read image: {image_path}")
        if gt is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        gt_resized = cv2.resize(gt, (width, height), interpolation=cv2.INTER_NEAREST) > 127
        model_input = resized.astype(np.float32) / 255.0

        interpreter.set_tensor(input_detail["index"], quantize_input(model_input[np.newaxis, ...], input_detail))
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        logits = dequantize_output(output, output_detail)[0, ..., 0]
        pred = sigmoid(logits) > args.threshold
        dice = dice_score(pred, gt_resized)

        gt_panel = colorize_mask(gt_resized, (0, 220, 255))
        pred_panel = colorize_mask(pred, (255, 80, 80))
        overlay = overlay_mask(resized, gt_resized, (0, 220, 255))
        overlay = overlay_mask(overlay, pred, (255, 80, 80))

        panels = [
            add_title(resized, "image"),
            add_title(gt_panel, "ground truth"),
            add_title(pred_panel, f"prediction dice={dice:.3f}"),
            add_title(overlay, "overlay: gt yellow, pred red"),
        ]
        composite = np.concatenate(panels, axis=1)
        out_path = out_dir / f"{image_path.stem}_viz.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))
        written.append((out_path, dice))

    for out_path, dice in written:
        print(f"{out_path} dice={dice:.4f}")


if __name__ == "__main__":
    main()
