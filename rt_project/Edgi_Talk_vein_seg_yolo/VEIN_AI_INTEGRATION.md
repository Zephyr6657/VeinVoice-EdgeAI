# Edgi-Talk Vein AI Integration

This stage adds board-side vein segmentation:

```text
USB UVC 320x240 YUYV
-> Tiny U-Net 160x160 int8 TFLite compiled by Vela
-> Ethos-U55 inference
-> blue-gray vein overlay on LCD preview
```

## Model

Source checkpoint:

```text
runs/tiny_unet/data1_final_v2_light_seed42_160_bc8_best.pth
```

Generated deployment models:

```text
runs/tiny_unet/data1_final_v2_light_160_int8.tflite
runs/vela_u55_160_final/data1_final_v2_light_160_int8_vela.tflite
```

The Vela model was embedded into:

```text
libraries/Common/deepcraft_ai/model/model.c
libraries/Common/deepcraft_ai/model/model.h
```

Vela result:

```text
CPU operators = 0
NPU operators = 44
Total SRAM used = 1200.25 KiB
Total Off-chip Flash used = 113.45 KiB
```

The board integration reserves a `1280 KiB` tensor arena in `.cy_socmem_data`.
The model input buffer is placed in shared HyperRAM to avoid overflowing the SOCMEM data region.

## Runtime Flow

The UVC frame loop is still in:

```text
libraries/Common/board/ports/usb/usbh_uvc_app.c
```

Frame path:

```text
uvc_ai_app_process_frame(frame)
uvc_display_frame(frame, stream_w, stream_h)
```

`uvc_ai_app_process_frame()` runs the model every `UVC_AI_FRAMES_TO_SKIP + 1` frames and keeps the latest mask.

`uvc_display_frame()` draws the camera frame first, then calls the overlay hook. The overlay hook blends mask pixels with a low-alpha blue-gray vein color.

## Important Parameters

```text
Camera input: 320x240 YUYV
Model input: int8[1,160,160,3]
Model output: int8[1,160,160,1]
Recommended threshold: 0.60
Board threshold_q: 0
Overlay color: RGB(35,80,95)
```

## Build Changes

- `BSP_USING_DEEPCRAFT_AI` is enabled in `rtconfig.h`.
- `libraries/Common/deepcraft_ai/SConscript` is added to `SConstruct`.
- USB port include path now includes `libraries/Common/deepcraft_ai/include`.

## Board Test

1. Build and flash this project in RT-Thread Studio.
2. Plug in the UVC camera.
3. Press `SW2/User Button` to start preview.
4. Expected logs:

```text
Vein AI model initialized input=160x160 threshold_q=0
Vein AI inference ... ms prep ... ms active=...
```

5. LCD should show camera preview with blue-gray vein overlay.

## Notes

This first board integration uses a simple display overlay. The aggressive PC-side visual completion is not included yet. After this stage runs on hardware, the next step can add lightweight connected-vein post-processing on the 160x160 mask.
