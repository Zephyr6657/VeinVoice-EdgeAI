# Edgi-Talk USB Camera LCD Preview

This project is the first board-side preview stage:

```text
USB UVC camera -> YUYV 320x240 -> RGB565 -> Edgi-Talk LCD
```

No model inference is included in this stage.

## What Changed

- The existing CherryUSB UVC example remains the base project.
- `usbh_uvc_app.c` now exposes:
  - `usbh_uvc_preview_is_ready()`
  - `usbh_uvc_preview_start(fmt, width, height)`
- `applications/main.c` starts a small `cam_auto` thread after USB host initialization.
- The thread waits until the UVC camera is enumerated, then waits for `SW2/User Button`.
- Pressing `SW2/User Button` starts:

```text
usbh_uvc_start 0 320 240
```

where `0` means YUYV.

The preview is a live camera stream after the button press. It is not a frozen single photo yet.

## Display Path

The project already includes `usbh_uvc_display.c`.

For a 320x240 YUYV camera frame, the display path is:

1. Convert YUYV to RGB565.
2. Scale the image to the LCD width.
3. Keep aspect ratio.
4. Center it vertically on the LCD with black top/bottom bands.
5. Flush the active LCD band.

## Build And Run

1. Open this project in RT-Thread Studio:

```text
官方USB摄像头例程目录/Edgi_Talk_M55_USB_UVC_1
```

2. Build and flash the M55 firmware.
3. Plug in the driverless USB camera.
4. Wait until the UVC device is detected.
5. Press `SW2/User Button` to start preview.

Expected serial log:

```text
Camera preview: waiting for UVC camera...
UVC device connected
Camera preview: camera ready, press SW2/User Button to start 320x240 YUYV
Camera preview: button pressed, start preview
UVC start: 320x240 format=yuyv
LCD: 512x800 16bpp ...
frame #1 size=153600 fmt=yuyv
```

## Manual Fallback

If the camera does not support 320x240 YUYV, use MSH to test a supported mode from the enumeration log:

```text
usbh_uvc_start 0 320 240
```

or MJPEG:

```text
usbh_uvc_start 1 320 240
```

If the stream is already running:

```text
usbh_uvc_stop
```

then start again with different parameters.

## Next Stage Entry Point

The model inference hook should be inserted between frame dequeue and LCD display in:

```text
libraries/Common/board/ports/usb/usbh_uvc_app.c
```

Current frame loop:

```text
usbh_video_stream_dequeue(...)
uvc_ai_app_process_frame(frame)
uvc_display_frame(frame, stream_w, stream_h)
usbh_video_stream_enqueue(frame)
```

The next phase can replace or extend `uvc_ai_app_process_frame(frame)` to run Tiny U-Net and render the processed vein display.
