# Hand Vein Segmentation

This repository currently uses Ultralytics YOLOv8n-seg for RTX 4060 training and
PC-side validation/demo. The YOLO checkpoint is not the final Ethos-U55 deployment
model.

The intended long-term flow is:

1. Train YOLOv8n-seg on PC to get a strong `best.pt`.
2. Use `best.pt` for pseudo-mask generation, data screening, and PC demos.
3. Train a Tiny U-Net / Mobile U-Net branch on PNG masks at 128x128 or 160x160.
4. Export the board model to full-int8 TFLite.
5. Compile the int8 TFLite model with Arm Vela for Ethos-U55.

## Deployment Boundary

YOLOv8n-seg is kept for PC training and validation only. It is not recommended as
the direct Ethos-U55 deployment target because YOLO segmentation normally brings
YOLO decode, NMS, dynamic output handling, and heavier post-processing.

Board-oriented models should follow these constraints:

- Prefer input sizes `128x128` or `160x160`.
- Prefer basic operators: `Conv2D`, `DepthwiseConv2D`, `ReLU`, `Pooling`,
  `ResizeNearestNeighbor`, `Concat`.
- Avoid YOLO decode, NMS, dynamic shapes, custom operators, and complex
  post-processing.
- Support full-int8 quantization.
- Export final deployment artifacts as int8 TFLite and compile them with Arm Vela.

## Environment

Recommended:

- Windows 10/11
- Python 3.10
- NVIDIA RTX 4060
- Recent NVIDIA driver with CUDA support

Python 3.13 is not recommended because some deep learning packages may lag behind
it.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For RTX 4060, install a CUDA-enabled PyTorch build if `torch.cuda.is_available()`
is `False` after installing requirements. Follow the selector on:

https://pytorch.org/get-started/locally/

Quick CUDA check:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Dataset Layout

Put the dataset in YOLOv8 segmentation format:

```text
datasets/hand_vein/
|-- train/images
|-- train/labels
|-- valid/images
|-- valid/labels
|-- test/images
|-- test/labels
`-- data.yaml
```

`datasets/hand_vein/data.yaml` should contain:

```yaml
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['vein']
```

Each label file must have the same stem as its image file. For example:

```text
train/images/sample_001.jpg
train/labels/sample_001.txt
```

YOLOv8 segmentation labels use polygon coordinates:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Coordinates must be normalized to `[0, 1]`. Detection box labels like
`class_id x_center y_center width height` are not valid for segmentation training.

## Check Dataset

Run this before training:

```powershell
python check_dataset.py
```

The checker verifies:

- `train`, `valid`, and `test` image/label directories exist
- every image has a same-name `.txt` label
- every label has a matching image
- labels are polygon segmentation rows, not detection box rows
- class ids and normalized coordinates are valid

## Train YOLOv8n-seg

```powershell
python train.py
```

Training uses:

- model: `yolov8n-seg.pt`
- data: `datasets/hand_vein/data.yaml`
- epochs: `100`
- imgsz: `640`
- batch: `8`
- device: `0`
- optimizer: `AdamW`
- lr0: `0.001`
- cos_lr: `True`
- patience: `30`
- amp: `True`
- close_mosaic: `10`

The expected best model path is:

```text
runs/segment/hand_vein_yolov8n/weights/best.pt
```

## PC Prediction Demo

After YOLO training:

```powershell
python predict.py
```

This loads `runs/segment/hand_vein_yolov8n/weights/best.pt`, runs inference on
`datasets/hand_vein/test/images`, and saves Ultralytics visualization/results
under:

```text
runs/segment/hand_vein_yolov8n_predict
```

## Generate Pseudo Masks

Use `best.pt` to batch-generate binary PNG masks:

```powershell
python predict_masks.py --source datasets/hand_vein/test/images --out datasets/hand_vein/test/pseudo_masks
```

Useful options:

```powershell
python predict_masks.py --source datasets/hand_vein/train/images --out datasets/hand_vein/train/pseudo_masks --conf 0.25 --overwrite
```

This produces one `*.png` mask per image. These masks can be used for data
screening or as pseudo-labels for Tiny U-Net / Mobile U-Net experiments.

If ground-truth YOLO polygon labels are available, convert them to binary masks:

```powershell
python yolo_polygon_to_mask.py --dataset datasets/hand_vein --splits train valid test --overwrite
```

This writes masks under each split's `masks/` folder.

## Train Tiny U-Net Branch

Tiny U-Net is the first lightweight segmentation branch for future deployment
experiments:

```powershell
python train_tiny_unet.py --imgsz 160 --batch 8 --epochs 40
```

For the Ethos-U55 direction, prefer:

```powershell
python train_tiny_unet.py --imgsz 128
python train_tiny_unet.py --imgsz 160
```

The current Tiny U-Net uses nearest-neighbor upsampling plus convolution instead
of transposed convolution, keeping the operator set closer to TFLite Micro / Vela
friendly deployment.

## Export YOLO ONNX

YOLO ONNX export is kept for PC-side inspection and interoperability, not as the
preferred Ethos-U55 deployment path:

```powershell
python export.py
```

The ONNX model is exported next to `best.pt`.

## Future Ethos-U55 Export Work

The board deployment branch now starts from the `128x128` Tiny U-Net checkpoint:

```text
runs/tiny_unet/tiny_unet_128_bc8_best.pth
```

### Export Fixed-Shape ONNX

```powershell
python export_tiny_unet_onnx.py --weights runs/tiny_unet/tiny_unet_128_bc8_best.pth --out runs/tiny_unet/tiny_unet_128_bc8.onnx
```

Validate ONNX against the PyTorch checkpoint:

```powershell
python validate_tiny_unet_onnx.py --weights runs/tiny_unet/tiny_unet_128_bc8_best.pth --onnx runs/tiny_unet/tiny_unet_128_bc8.onnx --image datasets/hand_vein/test/images
```

Expected shape:

```text
input:  image float32[1, 3, 128, 128]
output: logits float32[1, 1, 128, 128]
```

### Export Full-Int8 TFLite

The representative dataset is built from the existing test images and saved as
NHWC float32 samples:

```powershell
python export_tiny_unet_tflite_int8.py --weights runs/tiny_unet/tiny_unet_128_bc8_best.pth --test-images datasets/hand_vein/test/images --representative-out runs/tiny_unet/representative_test_128.npy --saved-model-dir runs/tiny_unet/tiny_unet_128_bc8_saved_model --out runs/tiny_unet/tiny_unet_128_bc8_int8.tflite
```

This exports:

```text
runs/tiny_unet/representative_test_128.npy
runs/tiny_unet/tiny_unet_128_bc8_saved_model/
runs/tiny_unet/tiny_unet_128_bc8_int8.tflite
```

Expected TFLite interface:

```text
input:  int8[1, 128, 128, 3]
output: int8[1, 128, 128, 1]
```

Validate the int8 TFLite model on the test split:

```powershell
python validate_tiny_unet_tflite.py --model runs/tiny_unet/tiny_unet_128_bc8_int8.tflite --dataset datasets/hand_vein --split test
```

Current PC-side int8 TFLite test result:

```text
images: 68
mean_dice: 0.8395
min_dice: 0.7393
max_dice: 0.9027
```

Remaining board deployment work:

- Board runtime integration.
- Input preprocessing and output mask post-processing on target.
- End-to-end latency, RAM, flash, and mask-quality testing on real hardware.

### Compile With Arm Vela

Install Vela if needed:

```powershell
python -m pip install ethos-u-vela
```

Compile for an Ethos-U55-128 target:

```powershell
python -m ethosu.vela runs/tiny_unet/tiny_unet_128_bc8_int8.tflite --accelerator-config ethos-u55-128 --optimise Performance --show-cpu-operations --verbose-performance --output-dir runs/vela_u55_128
```

Current Vela output:

```text
runs/vela_u55_128/tiny_unet_128_bc8_int8_vela.tflite
runs/vela_u55_128/tiny_unet_128_bc8_int8_summary_Ethos_U55_High_End_Embedded.csv
runs/vela_u55_128/tiny_unet_128_bc8_int8_per-layer.csv
```

Current Vela summary with default `Ethos_U55_High_End_Embedded` /
`Shared_Sram` settings:

```text
CPU operators: 0
NPU operators: 44
Total SRAM used: 768.28 KiB
Total off-chip flash used: 110.73 KiB
Estimated inference time: 0.005235428 s
Estimated throughput: 191.01 inferences/s
NN MACs: 147,947,520
```

Vela 5.0.0 depends on `flatbuffers==24.3.25`, while TensorFlow 2.21 requests a
newer `flatbuffers`. If TensorFlow export later reports a dependency issue,
either reinstall TensorFlow's preferred `flatbuffers` for export or keep Vela in
a separate environment.

Do not route YOLOv8n-seg directly into the Ethos-U55 deployment pipeline unless a
separate compatibility investigation proves that the decode/NMS/post-processing
path is acceptable.

## Common Issues

### CUDA is not available

Check the PyTorch CUDA build:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

If it prints `False`, install the CUDA-enabled PyTorch wheel from the official
PyTorch install selector.

### Out of memory on RTX 4060

The default YOLO config is tuned for RTX 4060:

```python
batch=8
imgsz=640
```

If GPU memory is insufficient, edit `train.py`:

```python
batch=4
```

If it is still insufficient, reduce image size:

```python
imgsz=512
```

### Label format error

If `check_dataset.py` reports a line with 5 values, the label is likely detection
format:

```text
class_id x_center y_center width height
```

YOLOv8 segmentation needs polygon points:

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

Re-export the dataset as YOLOv8 segmentation, not object detection.

### data.yaml path error

Because `data.yaml` lives inside `datasets/hand_vein`, relative paths should be:

```yaml
train: train/images
val: valid/images
test: test/images
```

Avoid `../train/images` unless your dataset is intentionally arranged outside
`datasets/hand_vein`.
