from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "runs" / "tiny_unet" / "tiny_unet_128_bc8_int8.tflite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate int8 Tiny U-Net TFLite on test masks.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--dataset", default="datasets/hand_vein")
    parser.add_argument("--split", default="test")
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


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    dataset_root = Path(args.dataset).resolve()
    images_dir = dataset_root / args.split / "images"
    masks_dir = dataset_root / args.split / "masks"

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

    scores = []
    for image_path in image_paths(images_dir):
        mask_path = masks_dir / f"{image_path.stem}.png"
        if not mask_path.is_file():
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        target = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 127

        interpreter.set_tensor(input_detail["index"], quantize_input(image[np.newaxis, ...], input_detail))
        interpreter.invoke()
        output = interpreter.get_tensor(output_detail["index"])
        logits = dequantize_output(output, output_detail)[0, ..., 0]
        pred = sigmoid(logits) > 0.5
        scores.append(dice_score(pred, target))

    print(f"model: {model_path}")
    print(f"split: {args.split}, images: {len(scores)}")
    print(f"mean_dice: {float(np.mean(scores)):.4f}")
    print(f"min_dice: {float(np.min(scores)):.4f}")
    print(f"max_dice: {float(np.max(scores)):.4f}")
    print(f"input: {input_detail['dtype'].__name__}{input_detail['shape'].tolist()} quant={input_detail['quantization']}")
    print(f"output: {output_detail['dtype'].__name__}{output_detail['shape'].tolist()} quant={output_detail['quantization']}")


if __name__ == "__main__":
    main()
