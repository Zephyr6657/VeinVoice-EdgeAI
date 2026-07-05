from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
import torch


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "tiny_unet" / "tiny_unet_128_bc8_best.pth"
DEFAULT_REP = ROOT / "runs" / "tiny_unet" / "representative_test_128.npy"
DEFAULT_OUT = ROOT / "runs" / "tiny_unet" / "tiny_unet_128_bc8_int8.tflite"
DEFAULT_SAVED_MODEL = ROOT / "runs" / "tiny_unet" / "tiny_unet_128_bc8_saved_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Tiny U-Net to full-int8 TFLite.")
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--test-images", default="datasets/hand_vein/test/images")
    parser.add_argument("--representative-out", default=str(DEFAULT_REP))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--saved-model-dir", default=str(DEFAULT_SAVED_MODEL))
    parser.add_argument("--max-representative", type=int, default=68)
    return parser.parse_args()


def conv_block(x, channels: int, name: str):
    x = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=True, name=f"{name}_conv1")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu1")(x)
    x = tf.keras.layers.Conv2D(channels, 3, padding="same", use_bias=True, name=f"{name}_conv2")(x)
    x = tf.keras.layers.ReLU(name=f"{name}_relu2")(x)
    return x


def resize_nearest(x, size: int, name: str):
    return tf.keras.layers.Resizing(size, size, interpolation="nearest", name=name)(x)


def build_model(imgsz: int, base_channels: int) -> tf.keras.Model:
    c = base_channels
    image = tf.keras.Input(shape=(imgsz, imgsz, 3), batch_size=1, name="image")
    e1 = conv_block(image, c, "enc1")
    e2 = conv_block(tf.keras.layers.MaxPooling2D(2, name="pool1")(e1), c * 2, "enc2")
    e3 = conv_block(tf.keras.layers.MaxPooling2D(2, name="pool2")(e2), c * 4, "enc3")
    b = conv_block(tf.keras.layers.MaxPooling2D(2, name="pool3")(e3), c * 8, "bottleneck")

    u3 = resize_nearest(b, imgsz // 4, "up3_resize")
    u3 = tf.keras.layers.Conv2D(c * 4, 1, padding="same", use_bias=True, name="up3_conv")(u3)
    d3 = conv_block(tf.keras.layers.Concatenate(axis=-1, name="cat3")([u3, e3]), c * 4, "dec3")

    u2 = resize_nearest(d3, imgsz // 2, "up2_resize")
    u2 = tf.keras.layers.Conv2D(c * 2, 1, padding="same", use_bias=True, name="up2_conv")(u2)
    d2 = conv_block(tf.keras.layers.Concatenate(axis=-1, name="cat2")([u2, e2]), c * 2, "dec2")

    u1 = resize_nearest(d2, imgsz, "up1_resize")
    u1 = tf.keras.layers.Conv2D(c, 1, padding="same", use_bias=True, name="up1_conv")(u1)
    d1 = conv_block(tf.keras.layers.Concatenate(axis=-1, name="cat1")([u1, e1]), c, "dec1")
    logits = tf.keras.layers.Conv2D(1, 1, padding="same", use_bias=True, name="head")(d1)
    return tf.keras.Model(image, logits, name="tiny_unet")


def set_conv(layer: tf.keras.layers.Layer, state: dict[str, torch.Tensor], key: str) -> None:
    weight = state[f"{key}.weight"].detach().cpu().numpy()
    values = [np.transpose(weight, (2, 3, 1, 0))]
    bias_key = f"{key}.bias"
    if bias_key in state:
        values.append(state[bias_key].detach().cpu().numpy())
    else:
        values.append(np.zeros(weight.shape[0], dtype=np.float32))
    layer.set_weights(values)


def set_fused_conv_bn(
    layer: tf.keras.layers.Layer,
    state: dict[str, torch.Tensor],
    conv_key: str,
    bn_key: str,
    eps: float = 1e-5,
) -> None:
    weight = state[f"{conv_key}.weight"].detach().cpu().numpy()
    gamma = state[f"{bn_key}.weight"].detach().cpu().numpy()
    beta = state[f"{bn_key}.bias"].detach().cpu().numpy()
    mean = state[f"{bn_key}.running_mean"].detach().cpu().numpy()
    var = state[f"{bn_key}.running_var"].detach().cpu().numpy()

    scale = gamma / np.sqrt(var + eps)
    fused_weight = weight * scale[:, None, None, None]
    fused_bias = beta - mean * scale
    layer.set_weights([np.transpose(fused_weight, (2, 3, 1, 0)), fused_bias])


def set_block(model: tf.keras.Model, state: dict[str, torch.Tensor], keras_name: str, torch_name: str) -> None:
    set_fused_conv_bn(
        model.get_layer(f"{keras_name}_conv1"),
        state,
        f"{torch_name}.block.0",
        f"{torch_name}.block.1",
    )
    set_fused_conv_bn(
        model.get_layer(f"{keras_name}_conv2"),
        state,
        f"{torch_name}.block.3",
        f"{torch_name}.block.4",
    )


def load_torch_weights(model: tf.keras.Model, checkpoint: dict) -> None:
    state = checkpoint["model"]
    for name in ("enc1", "enc2", "enc3", "bottleneck", "dec3", "dec2", "dec1"):
        set_block(model, state, name, name)
    set_conv(model.get_layer("up3_conv"), state, "up3_conv")
    set_conv(model.get_layer("up2_conv"), state, "up2_conv")
    set_conv(model.get_layer("up1_conv"), state, "up1_conv")
    set_conv(model.get_layer("head"), state, "head")


def image_paths(images_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def load_representative_images(images_dir: Path, imgsz: int, limit: int) -> np.ndarray:
    paths = image_paths(images_dir)[:limit]
    if not paths:
        raise FileNotFoundError(f"No calibration images found in {images_dir}")

    samples = []
    for path in paths:
        data = np.fromfile(str(path), dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        samples.append(image.astype(np.float32) / 255.0)
    return np.stack(samples, axis=0)


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights).resolve()
    test_images = Path(args.test_images).resolve()
    rep_path = Path(args.representative_out).resolve()
    out_path = Path(args.out).resolve()
    saved_model_dir = Path(args.saved_model_dir).resolve()

    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")
    if not test_images.is_dir():
        raise FileNotFoundError(f"Test image directory not found: {test_images}")

    checkpoint = torch.load(weights_path, map_location="cpu")
    imgsz = int(checkpoint["imgsz"])
    base_channels = int(checkpoint["base_channels"])

    model = build_model(imgsz, base_channels)
    load_torch_weights(model, checkpoint)

    representative = load_representative_images(test_images, imgsz, args.max_representative)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(rep_path, representative)

    def representative_dataset():
        for sample in representative:
            yield [sample[np.newaxis, ...].astype(np.float32)]

    saved_model_dir.mkdir(parents=True, exist_ok=True)
    model.export(str(saved_model_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    print(f"Saved representative data: {rep_path} shape={representative.shape} dtype=float32")
    print(f"Saved TensorFlow model: {saved_model_dir}")
    print(f"Exported full-int8 TFLite: {out_path}")
    print(f"input:  int8[1, {imgsz}, {imgsz}, 3]")
    print(f"output: int8[1, {imgsz}, {imgsz}, 1]")


if __name__ == "__main__":
    main()
