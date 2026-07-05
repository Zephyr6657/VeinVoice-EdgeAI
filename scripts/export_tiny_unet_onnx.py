from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

from train_tiny_unet import TinyUNet


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "tiny_unet" / "tiny_unet_128_bc8_best.pth"
DEFAULT_OUT = ROOT / "runs" / "tiny_unet" / "tiny_unet_128_bc8.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Tiny U-Net checkpoint to fixed-shape ONNX.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--opset", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    out_path = Path(args.out).resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    imgsz = int(checkpoint["imgsz"])
    base_channels = int(checkpoint["base_channels"])

    model = TinyUNet(base_channels=base_channels)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    onnx_model = onnx.load(str(out_path))
    onnx.checker.check_model(onnx_model)

    print(f"Exported ONNX: {out_path}")
    print(f"input:  image float32[1, 3, {imgsz}, {imgsz}]")
    print(f"output: logits float32[1, 1, {imgsz}, {imgsz}]")


if __name__ == "__main__":
    main()
