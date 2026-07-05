from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch

from train_tiny_unet import TinyUNet


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "tiny_unet" / "tiny_unet_128_bc8_best.pth"
DEFAULT_ONNX = ROOT / "runs" / "tiny_unet" / "tiny_unet_128_bc8.onnx"
DEFAULT_IMAGE = ROOT / "datasets" / "hand_vein" / "test" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare Tiny U-Net PyTorch and ONNX outputs.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--image", default=str(DEFAULT_IMAGE))
    return parser.parse_args()


def first_image(path: Path) -> Path:
    if path.is_file():
        return path
    for image_path in sorted(path.iterdir()):
        if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}:
            return image_path
    raise FileNotFoundError(f"No image found in {path}")


def load_image(path: Path, imgsz: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    nhwc = image.astype(np.float32) / 255.0
    nchw = np.transpose(nhwc, (2, 0, 1))[np.newaxis, ...]
    return nhwc[np.newaxis, ...], nchw


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    onnx_path = Path(args.onnx).resolve()
    image_path = first_image(Path(args.image).resolve())

    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    imgsz = int(checkpoint["imgsz"])
    base_channels = int(checkpoint["base_channels"])
    _, nchw = load_image(image_path, imgsz)

    model = TinyUNet(base_channels=base_channels)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        torch_output = model(torch.from_numpy(nchw)).numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"image": nchw})[0]
    abs_diff = np.abs(torch_output - onnx_output)

    print(f"image: {image_path}")
    print(f"onnx: {onnx_path}")
    print(f"torch_shape: {list(torch_output.shape)}")
    print(f"onnx_shape: {list(onnx_output.shape)}")
    print(f"max_abs_diff: {float(abs_diff.max()):.8f}")
    print(f"mean_abs_diff: {float(abs_diff.mean()):.8f}")


if __name__ == "__main__":
    main()
