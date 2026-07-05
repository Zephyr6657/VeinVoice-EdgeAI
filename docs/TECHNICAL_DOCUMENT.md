# 手背静脉图像分割项目技术文档

## 1. 项目概述

本项目面向手背静脉图像的语义分割任务，目标是在 PC 端完成模型训练、验证和可视化，并将最终轻量化模型部署到 Arm Ethos-U55 NPU 平台。项目采用“两阶段模型路线”：首先使用 YOLOv8n-seg 作为 PC 端高精度分割模型，用于数据验证、伪标签生成和效果演示；随后训练面向嵌入式部署的 Tiny U-Net 模型，并完成 ONNX、full-int8 TFLite 和 Vela 编译产物导出。

最终交付模型为 `128x128` 输入的 full-int8 Tiny U-Net TFLite 模型，经 Arm Vela 编译后可在 Ethos-U55 上运行。YOLOv8n-seg 不作为板端部署模型，仅作为训练辅助和 PC 端验证工具。

## 2. 技术目标

- 构建手背静脉单类别分割数据集，类别为 `vein`。
- 完成 YOLOv8n-seg 分割模型训练，用于 PC 端验证和伪 mask 生成。
- 将 YOLO polygon 标注和 Labelme polygon 标注转换为二值 PNG mask。
- 训练 Tiny U-Net 轻量化语义分割模型。
- 基于小规模新增标注数据完成 Tiny U-Net 微调。
- 导出固定输入尺寸 ONNX 模型，并与 PyTorch 结果对齐验证。
- 导出 full-int8 TFLite 模型，并在 PC 端验证量化模型精度。
- 使用 Arm Vela 编译 TFLite 模型，生成 Ethos-U55 部署文件。
- 完成板端前处理、推理、后处理和端到端性能测试。

## 3. 运行环境

### 3.1 开发环境

- 操作系统：Windows 10/11
- Python：3.10
- GPU：NVIDIA RTX 4060
- CUDA：与本地 PyTorch 版本匹配
- 主要深度学习框架：PyTorch、Ultralytics YOLO、TensorFlow/TFLite

### 3.2 Python 依赖

项目核心依赖位于 `requirements.txt`：

```text
ultralytics>=8.2.0
torch
torchvision
onnx
onnxsim
onnxruntime
tensorflow
```

建议使用虚拟环境：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

若用于训练的 PyTorch 需要 CUDA 支持，应安装与显卡和驱动匹配的 CUDA 版 PyTorch。

## 4. 项目目录结构

```text
vein_seg_yolo/
|-- datasets/
|   |-- hand_vein/                 # 主训练数据集，YOLO 标签和 mask 数据
|   `-- finetune_hand_vein/        # 小样本微调数据集
|-- my_test_images/                # 外部测试/实验数据
|-- runs/
|   |-- segment/                   # YOLO 训练、验证、预测输出
|   |-- tiny_unet/                 # Tiny U-Net 权重、导出和可视化结果
|   `-- vela_u55_128/              # Vela 编译产物
|-- 数据集/                         # Labelme 原始标注数据
|-- check_dataset.py               # YOLO 分割数据集检查
|-- train.py                       # YOLOv8n-seg 训练入口
|-- predict.py                     # YOLO PC 端预测演示
|-- predict_masks.py               # YOLO 伪 mask 批量生成
|-- yolo_polygon_to_mask.py        # YOLO polygon 转 PNG mask
|-- labelme_to_masks.py            # Labelme polygon 转 PNG mask
|-- train_tiny_unet.py             # Tiny U-Net 训练
|-- finetune_tiny_unet.py          # Tiny U-Net 小样本微调
|-- export_tiny_unet_onnx.py       # Tiny U-Net ONNX 导出
|-- validate_tiny_unet_onnx.py     # ONNX 对齐验证
|-- export_tiny_unet_tflite_int8.py# full-int8 TFLite 导出
|-- validate_tiny_unet_tflite.py   # TFLite 精度验证
|-- visualize_tiny_unet_pytorch.py # PyTorch 模型可视化
`-- visualize_tiny_unet_tflite.py  # TFLite 模型可视化
```

## 5. 数据集设计

### 5.1 主数据集

主数据集位于 `datasets/hand_vein`，采用 YOLOv8 segmentation 格式，同时包含由 polygon 标签转换得到的二值 mask。

```text
datasets/hand_vein/
|-- train/images
|-- train/labels
|-- train/masks
|-- valid/images
|-- valid/labels
|-- valid/masks
|-- test/images
|-- test/labels
|-- test/masks
`-- data.yaml
```

当前数据规模：

| 划分 | 图像数量 | 用途 |
| --- | ---: | --- |
| train | 1539 | 模型训练 |
| valid | 155 | 训练过程验证和模型选择 |
| test | 68 | 最终测试、TFLite 量化校准和精度评估 |

`data.yaml` 配置为：

```yaml
train: train/images
val: valid/images
test: test/images

nc: 1
names: ['vein']
```

### 5.2 微调数据集

微调数据集位于 `datasets/finetune_hand_vein`，使用扁平结构：

```text
datasets/finetune_hand_vein/
|-- images
`-- masks
```

该数据集由 Labelme 标注转换得到，共 46 张图像，主要用于改善模型在新增采集场景下的泛化能力。

### 5.3 标注格式

YOLO 分割标签使用 polygon 归一化坐标：

```text
class_id x1 y1 x2 y2 x3 y3 ...
```

其中 `class_id=0` 表示静脉区域，所有坐标归一化到 `[0, 1]`。检测框格式 `class_id x_center y_center width height` 不适用于本项目分割训练。

模型训练和部署使用的 mask 为单通道二值 PNG：

- 背景：0
- 静脉：255

## 6. 数据处理流程

### 6.1 数据集检查

使用 `check_dataset.py` 对 YOLO 数据集进行一致性检查：

```powershell
python check_dataset.py
```

检查内容包括：

- `train`、`valid`、`test` 目录是否完整。
- 每张图像是否存在同名 `.txt` 标签。
- 每个标签是否有匹配图像。
- 标签是否为 segmentation polygon 格式。
- 类别 id 和归一化坐标是否合法。

### 6.2 YOLO polygon 转 mask

将 YOLO segmentation 标签转换为 Tiny U-Net 所需的 PNG mask：

```powershell
python yolo_polygon_to_mask.py --dataset datasets/hand_vein --splits train valid test --overwrite
```

输出路径为各划分目录下的 `masks/`。

### 6.3 Labelme 标注转 mask

新增样本采用 Labelme 标注，使用 `labelme_to_masks.py` 转换：

```powershell
python labelme_to_masks.py --input 数据集 --out datasets/finetune_hand_vein --label vein --overwrite
```

转换后会复制原图到 `images/`，并将 `vein` polygon 填充为二值 mask 写入 `masks/`。

## 7. 模型方案

### 7.1 YOLOv8n-seg PC 端模型

YOLOv8n-seg 用于 PC 端训练、验证、预测展示和伪 mask 生成。训练入口为 `train.py`，主要配置如下：

| 参数 | 值 |
| --- | --- |
| 预训练权重 | `yolov8n-seg.pt` |
| 数据配置 | `datasets/hand_vein/data.yaml` |
| epoch | 100 |
| 输入尺寸 | 640 |
| batch | 8 |
| 优化器 | AdamW |
| 初始学习率 | 0.001 |
| 学习率策略 | cosine |
| patience | 30 |
| AMP | enabled |

训练命令：

```powershell
python train.py
```

最优权重输出：

```text
runs/segment/hand_vein_yolov8n/weights/best.pt
```

PC 端预测：

```powershell
python predict.py
```

伪 mask 生成：

```powershell
python predict_masks.py --source datasets/hand_vein/test/images --out datasets/hand_vein/test/pseudo_masks
```

YOLO 路线的优点是训练便利、可视化能力强、对复杂形态区域拟合能力好；缺点是其部署链路包含 decode、NMS、动态输出和较重后处理，不适合作为 Ethos-U55 的直接部署模型。

### 7.2 Tiny U-Net 板端模型

Tiny U-Net 是本项目最终部署模型。模型采用 encoder-decoder 结构，包含三层下采样、瓶颈层和三层上采样。上采样使用 nearest-neighbor resize 加卷积，而不是反卷积，以提高 TFLite Micro 和 Vela 兼容性。

模型结构特点：

- 输入：RGB 图像，固定尺寸 `128x128` 或 `160x160`。
- 输出：单通道静脉 logits，尺寸与输入一致。
- 基础通道数：默认 `base_channels=8`。
- 激活函数：ReLU。
- 下采样：MaxPool2D。
- 上采样：Nearest resize。
- skip connection：encoder 与 decoder 特征拼接。
- 损失函数：BCEWithLogits + Dice Loss。
- 评价指标：Dice score。

推荐板端配置：

```text
input:  float32[1, 3, 128, 128]   # PyTorch/ONNX
output: float32[1, 1, 128, 128]
```

量化部署接口：

```text
input:  int8[1, 128, 128, 3]      # TFLite
output: int8[1, 128, 128, 1]
```

## 8. 训练流程

### 8.1 Tiny U-Net 基础训练

训练命令：

```powershell
python train_tiny_unet.py --dataset datasets/hand_vein --imgsz 128 --base-channels 8 --batch 8 --epochs 40 --out runs/tiny_unet/tiny_unet_128_bc8_best.pth
```

训练过程使用：

- 随机水平翻转。
- 随机垂直翻转。
- AdamW 优化器。
- CosineAnnealingLR 学习率调度。
- CUDA AMP 混合精度训练。
- 验证集 Dice 最优模型保存。

### 8.2 小样本微调

基础模型训练完成后，使用新增 Labelme 标注数据进行微调：

```powershell
python finetune_tiny_unet.py --dataset datasets/finetune_hand_vein --weights runs/tiny_unet/tiny_unet_128_bc8_best.pth --out runs/tiny_unet/tiny_unet_128_bc8_finetuned_v2.pth --epochs 80 --batch 4 --lr 1e-4
```

微调流程会从 checkpoint 中读取 `imgsz` 和 `base_channels`，保持模型结构一致。默认按 `val-ratio=0.25` 划分训练/验证集，并增加亮度、对比度扰动以提升采集条件变化下的鲁棒性。

## 9. 模型导出与验证

### 9.1 ONNX 导出

固定形状 ONNX 导出命令：

```powershell
python export_tiny_unet_onnx.py --weights runs/tiny_unet/tiny_unet_128_bc8_best.pth --out runs/tiny_unet/tiny_unet_128_bc8.onnx
```

ONNX 输入输出：

```text
input:  image float32[1, 3, 128, 128]
output: logits float32[1, 1, 128, 128]
```

导出后使用 `validate_tiny_unet_onnx.py` 与 PyTorch checkpoint 进行数值对齐：

```powershell
python validate_tiny_unet_onnx.py --weights runs/tiny_unet/tiny_unet_128_bc8_best.pth --onnx runs/tiny_unet/tiny_unet_128_bc8.onnx --image datasets/hand_vein/test/images
```

### 9.2 full-int8 TFLite 导出

TFLite 导出流程在 `export_tiny_unet_tflite_int8.py` 中完成。该脚本会重建等价的 Keras Tiny U-Net，将 PyTorch 权重转换到 TensorFlow 模型，并融合 BatchNorm 参数到 Conv2D 权重中。

导出命令：

```powershell
python export_tiny_unet_tflite_int8.py --weights runs/tiny_unet/tiny_unet_128_bc8_best.pth --test-images datasets/hand_vein/test/images --representative-out runs/tiny_unet/representative_test_128.npy --saved-model-dir runs/tiny_unet/tiny_unet_128_bc8_saved_model --out runs/tiny_unet/tiny_unet_128_bc8_int8.tflite
```

输出文件：

```text
runs/tiny_unet/representative_test_128.npy
runs/tiny_unet/tiny_unet_128_bc8_saved_model/
runs/tiny_unet/tiny_unet_128_bc8_int8.tflite
```

量化策略：

- 使用测试集图像构建 representative dataset。
- 输入类型：int8。
- 输出类型：int8。
- 算子集合：`TFLITE_BUILTINS_INT8`。
- 模型输入预处理：RGB、resize 到 `128x128`、归一化到 `[0, 1]` 后按 TFLite input scale/zero point 量化。

### 9.3 TFLite 精度验证

验证命令：

```powershell
python validate_tiny_unet_tflite.py --model runs/tiny_unet/tiny_unet_128_bc8_int8.tflite --dataset datasets/hand_vein --split test
```

当前 PC 端 int8 TFLite 测试结果：

| 指标 | 数值 |
| --- | ---: |
| 测试图像数 | 68 |
| mean Dice | 0.8395 |
| min Dice | 0.7393 |
| max Dice | 0.9027 |

## 10. Ethos-U55 部署流程

### 10.1 Vela 编译

使用 Arm Vela 将 full-int8 TFLite 编译为 Ethos-U55 可执行图：

```powershell
python -m ethosu.vela runs/tiny_unet/tiny_unet_128_bc8_int8.tflite --accelerator-config ethos-u55-128 --optimise Performance --show-cpu-operations --verbose-performance --output-dir runs/vela_u55_128
```

输出文件：

```text
runs/vela_u55_128/tiny_unet_128_bc8_int8_vela.tflite
runs/vela_u55_128/tiny_unet_128_bc8_int8_summary_Ethos_U55_High_End_Embedded.csv
runs/vela_u55_128/tiny_unet_128_bc8_int8_per-layer.csv
```

当前 Vela 编译摘要：

| 项目 | 数值 |
| --- | ---: |
| CPU operators | 0 |
| NPU operators | 44 |
| Total SRAM used | 768.28 KiB |
| Total off-chip flash used | 110.73 KiB |
| Estimated inference time | 0.005235428 s |
| Estimated throughput | 191.01 inferences/s |
| NN MACs | 147,947,520 |

该结果表明模型全部算子均已映射到 NPU，没有 CPU fallback，满足 Ethos-U55 部署要求。

### 10.2 板端推理链路

板端推理流程如下：

1. 摄像头或上位机输入原始手背图像。
2. 将图像 resize 到 `128x128`。
3. 按训练时规则转换为 RGB。
4. 将像素归一化到 `[0, 1]`。
5. 根据 TFLite 输入量化参数转换为 int8。
6. 调用 Ethos-U55 推理。
7. 根据输出 scale/zero point 反量化 logits。
8. 执行 sigmoid。
9. 使用阈值 `0.5` 得到二值静脉 mask。
10. 根据业务需要进行形态学后处理、连通域过滤或映射回原图尺寸。

### 10.3 部署边界

板端部署只使用 Tiny U-Net full-int8 模型，不直接部署 YOLOv8n-seg。该边界可以避免：

- YOLO decode 后处理。
- NMS 算子或自定义后处理。
- 动态输出 tensor。
- 较大的模型参数和内存占用。
- TFLite Micro/Ethos-U 兼容性风险。

## 11. 可视化与结果检查

PyTorch 模型可视化：

```powershell
python visualize_tiny_unet_pytorch.py --weights runs/tiny_unet/tiny_unet_128_bc8_best.pth --dataset datasets/hand_vein --split test
```

TFLite 模型可视化：

```powershell
python visualize_tiny_unet_tflite.py --model runs/tiny_unet/tiny_unet_128_bc8_int8.tflite --dataset datasets/hand_vein --split test
```

可视化结果保存在 `runs/tiny_unet/` 下，用于比较原图、真实 mask 和预测 mask 的空间一致性。微调后模型的可视化结果保存在：

```text
runs/tiny_unet/finetune_visualizations/
runs/tiny_unet/finetune_v2_visualizations/
```

## 12. 关键工程实现

### 12.1 Tiny U-Net 数据读取

`VeinMaskDataset` 负责读取 `images/` 与 `masks/` 下的同名文件，并将图像转换为 `[C, H, W]` float tensor，将 mask 转换为 `[1, H, W]` 二值 tensor。

### 12.2 损失函数

训练损失由 BCE 和 Dice Loss 组成：

```text
Loss = BCEWithLogits(logits, mask) + DiceLoss(sigmoid(logits), mask)
```

BCE 保证像素级分类稳定性，Dice Loss 强化前景区域重叠质量，适合静脉这类细长目标区域。

### 12.3 PyTorch 到 TFLite 权重转换

TFLite 导出脚本没有直接从 ONNX 转换，而是构建等价 Keras 网络后加载 PyTorch 参数。卷积层权重从 PyTorch 的 `[out, in, kh, kw]` 转为 TensorFlow 的 `[kh, kw, in, out]`。BatchNorm 参数在导出前融合进 Conv2D：

```text
scale = gamma / sqrt(var + eps)
fused_weight = conv_weight * scale
fused_bias = beta - mean * scale
```

这样可以减少部署图中的 BatchNorm 算子，提高 TFLite/Vela 兼容性。

## 13. 性能与资源评估

当前推荐部署模型为：

```text
runs/vela_u55_128/tiny_unet_128_bc8_int8_vela.tflite
```

综合评估：

- 精度方面，int8 TFLite 在测试集上 mean Dice 达到 0.8395，能够较稳定地恢复主要静脉区域。
- 性能方面，Vela 估计单次推理约 5.24 ms，吞吐约 191 FPS。
- 资源方面，SRAM 使用约 768.28 KiB，off-chip flash 使用约 110.73 KiB。
- 兼容性方面，Vela 报告 CPU operators 为 0，说明图中算子均可由 NPU 执行。

## 14. 已完成交付物

| 类型 | 路径 |
| --- | --- |
| YOLO 训练权重 | `runs/segment/hand_vein_yolov8n/weights/best.pt` |
| Tiny U-Net PyTorch 权重 | `runs/tiny_unet/tiny_unet_128_bc8_best.pth` |
| Tiny U-Net 微调权重 | `runs/tiny_unet/tiny_unet_128_bc8_finetuned_v2.pth` |
| ONNX 模型 | `runs/tiny_unet/tiny_unet_128_bc8.onnx` |
| SavedModel | `runs/tiny_unet/tiny_unet_128_bc8_saved_model/` |
| full-int8 TFLite | `runs/tiny_unet/tiny_unet_128_bc8_int8.tflite` |
| Vela 编译模型 | `runs/vela_u55_128/tiny_unet_128_bc8_int8_vela.tflite` |
| Vela 性能摘要 | `runs/vela_u55_128/tiny_unet_128_bc8_int8_summary_Ethos_U55_High_End_Embedded.csv` |
| 可视化结果 | `runs/tiny_unet/test_visualizations/` |

## 15. 结论

本项目完成了从手背静脉图像标注、数据检查、YOLO PC 端分割训练、Tiny U-Net 轻量模型训练、微调、ONNX 导出、full-int8 TFLite 量化到 Ethos-U55 Vela 编译的完整工程闭环。

最终方案保留 YOLOv8n-seg 作为高精度 PC 端辅助模型，同时将 Tiny U-Net 作为板端部署模型。该设计兼顾了训练阶段的数据处理效率、分割质量和嵌入式端部署可行性。根据当前 TFLite 精度验证和 Vela 编译结果，模型已经满足 Ethos-U55 部署的算子兼容性、资源占用和实时性要求，可进入实际硬件集成、场景测试和产品化调优阶段。
