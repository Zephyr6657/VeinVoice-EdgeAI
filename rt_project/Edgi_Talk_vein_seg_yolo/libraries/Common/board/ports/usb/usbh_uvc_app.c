/*
 * CherryUSB UVC Host Application Layer
 * FPS monitoring, msh commands, video device callbacks, frame processing.
 */

#include <rtthread.h>
#include <rtdevice.h>
#include "usbh_uvc_stream.h"
#include "usb_config.h"

#define UVC_APP_ENABLE_AI 1

#if defined(BSP_USING_DEEPCRAFT_AI) && UVC_APP_ENABLE_AI
#include "../../../deepcraft_ai/include/uvc_ai_app.h"
#else
static void uvc_ai_app_enforce_mode(uint8_t *fmt, uint16_t *width, uint16_t *height)
{
    (void)fmt;
    (void)width;
    (void)height;
}

static int uvc_ai_app_start(uint16_t src_width, uint16_t src_height)
{
    (void)src_width;
    (void)src_height;
    return 0;
}

static int uvc_ai_app_process_frame(const struct usbh_videoframe *frame)
{
    (void)frame;
    return 0;
}

static void uvc_ai_app_clear_result(void)
{
}

static void uvc_ai_app_stop(void)
{
}
#endif

#undef  USB_DBG_TAG
#define USB_DBG_TAG "uvc_app"
#include "usb_log.h"

int uvc_display_init(void);
void uvc_display_frame(struct usbh_videoframe *frame, uint16_t src_w, uint16_t src_h);

/* ---------- frame buffers ---------- */

/*
 * Use one raw-sized backing store for both YUYV and MJPEG.
 * This keeps MJPEG safe even when the camera uses a larger-than-expected
 * compressed frame, while preserving the existing YUYV path.
 */
#define UVC_FRAME_BUF_SIZE  (256 * 1024)
#define UVC_FRAME_BUF_COUNT 2

static USB_NOCACHE_RAM_SECTION USB_MEM_ALIGNX uint8_t frame_buffer[UVC_FRAME_BUF_COUNT][UVC_FRAME_BUF_SIZE];
static struct usbh_videoframe frame_pool[UVC_FRAME_BUF_COUNT];

static volatile uint8_t uvc_app_running;
static volatile uint8_t uvc_device_ready;
static volatile uint8_t uvc_last_capture_ok;
static volatile uint8_t uvc_display_freeze;
static volatile uint8_t uvc_ai_process_request;
static volatile uint8_t uvc_ai_processing_busy;
static volatile uint8_t uvc_force_live_redraw;
static volatile uint32_t g_uvc_display_fps;
static volatile uint32_t g_uvc_display_drop_count;

static void uvc_filter_empty_formats(struct usbh_video *video_class)
{
    uint8_t read_idx;
    uint8_t write_idx;

    if (video_class == RT_NULL) {
        return;
    }

    /*
     * Some cameras expose an extra VS format entry with zero frames.
     * Keep CherryUSB core unchanged and compact valid entries here,
     * so usbh_video_open() does not pick an empty format.
     */
    write_idx = 0U;
    for (read_idx = 0U; read_idx < video_class->num_of_formats; read_idx++) {
        if (video_class->format[read_idx].num_of_frames == 0U) {
            continue;
        }

        if (write_idx != read_idx) {
            video_class->format[write_idx] = video_class->format[read_idx];
        }
        write_idx++;
    }

    if (write_idx != video_class->num_of_formats) {
        USB_LOG_WRN("UVC filter empty format: %u -> %u\r\n",
                    video_class->num_of_formats, write_idx);
        video_class->num_of_formats = write_idx;
    }
}

/* ---------- CherryUSB video class callbacks ---------- */

void usbh_video_run(struct usbh_video *video_class)
{
    uvc_filter_empty_formats(video_class);
    uvc_device_ready = 1;
    USB_LOG_INFO("UVC device connected\r\n");
    usbh_video_list_info(video_class);
}

void usbh_video_stop(struct usbh_video *video_class)
{
    USB_LOG_INFO("UVC device disconnected\r\n");
    usbh_video_stream_stop();
    uvc_app_running = 0;
    uvc_device_ready = 0;
}

/* ---------- FPS print thread ---------- */

extern volatile uint32_t g_uvc_fps;
extern volatile uint32_t uvc_transfer_count;

static void uvc_display_fps_record(void)
{
    static uint32_t frame_count;
    static rt_tick_t tick_last;
    rt_tick_t tick_now;
    rt_tick_t tick_delta;

    frame_count++;
    if ((frame_count % 10U) != 0U) {
        return;
    }

    tick_now = rt_tick_get();
    if (tick_last == 0U) {
        tick_last = tick_now;
        return;
    }

    tick_delta = tick_now - tick_last;
    if (tick_delta > 0U) {
        g_uvc_display_fps = (10U * RT_TICK_PER_SECOND) / tick_delta;
    }
    tick_last = tick_now;
}

/* ---------- frame consumer thread ---------- */

static uint16_t uvc_cam_w, uvc_cam_h;

static void uvc_frame_thread_entry(void *arg)
{
    struct usbh_videoframe *frame;
    uint32_t frame_count = 0;
    int ret;
    uint16_t stream_w;
    uint16_t stream_h;

    (void)arg;
    uvc_last_capture_ok = 0U;

    /* Initialize display output and show a visible capture-wait pattern. */
    if (uvc_display_init() != 0) {
        USB_LOG_ERR("LCD init failed, abort snapshot\r\n");
        goto snapshot_exit;
    }

#if UVC_APP_ENABLE_AI
    if (uvc_ai_app_start(uvc_cam_w, uvc_cam_h) != RT_EOK) {
        USB_LOG_WRN("AI init failed, snapshot will display raw camera frame\r\n");
    }
#else
    USB_LOG_INFO("AI disabled for UVC camera diagnostics\r\n");
#endif

    while (uvc_app_running) {
        ret = usbh_video_stream_dequeue(&frame, 5U * RT_TICK_PER_SECOND);
        if (ret < 0) {
            continue;
        }

        if (frame == RT_NULL) {
            continue;
        }

        frame_count++;

        if (frame_count < 3U) {
            usbh_video_stream_enqueue(frame);
            continue;
        }

        usbh_video_stream_get_info(&stream_w, &stream_h, RT_NULL);

        if (uvc_ai_process_request) {
            uvc_ai_processing_busy = 1U;
            uvc_ai_process_request = 0U;
            uvc_display_freeze = 1U;

#if UVC_APP_ENABLE_AI
            uvc_ai_app_clear_result();
#endif
            uvc_display_frame(frame,
                              stream_w ? stream_w : uvc_cam_w,
                              stream_h ? stream_h : uvc_cam_h);
#if UVC_APP_ENABLE_AI
            (void)uvc_ai_app_process_frame(frame);
            uvc_display_frame(frame,
                              stream_w ? stream_w : uvc_cam_w,
                              stream_h ? stream_h : uvc_cam_h);
#endif
            uvc_display_fps_record();
            uvc_ai_processing_busy = 0U;
        } else if ((!uvc_display_freeze) || uvc_force_live_redraw) {
            uvc_force_live_redraw = 0U;
            uvc_display_frame(frame,
                              stream_w ? stream_w : uvc_cam_w,
                              stream_h ? stream_h : uvc_cam_h);
            uvc_display_fps_record();
        }
        uvc_last_capture_ok = 1U;

        usbh_video_stream_enqueue(frame);
    }

snapshot_exit:
    usbh_video_stream_stop();
#if UVC_APP_ENABLE_AI
    uvc_ai_app_stop();
#endif
    uvc_app_running = 0U;
    USB_LOG_INFO("UVC preview thread exit frames=%u\r\n", (unsigned)frame_count);
}

/* ---------- public preview control ---------- */

int usbh_uvc_preview_is_ready(void)
{
    return uvc_device_ready ? 1 : 0;
}

int usbh_uvc_preview_is_running(void)
{
    return uvc_app_running ? 1 : 0;
}

int usbh_uvc_preview_last_capture_ok(void)
{
    return uvc_last_capture_ok ? 1 : 0;
}

int usbh_uvc_preview_toggle_freeze(void)
{
    if (uvc_ai_processing_busy) {
        USB_LOG_WRN("UVC AI snapshot ignored, AI busy\r\n");
        return uvc_display_freeze ? 1 : 0;
    }

    if (uvc_display_freeze) {
        uvc_display_freeze = 0U;
        uvc_ai_process_request = 0U;
        uvc_force_live_redraw = 1U;
#if UVC_APP_ENABLE_AI
        uvc_ai_app_clear_result();
#endif
        USB_LOG_INFO("UVC display live, old AI overlay cleared\r\n");
        return 0;
    }

    uvc_display_freeze = 1U;
    uvc_ai_process_request = 1U;
    uvc_force_live_redraw = 0U;
#if UVC_APP_ENABLE_AI
    uvc_ai_app_clear_result();
#endif
    USB_LOG_INFO("UVC AI snapshot requested\r\n");
    return 1;
}

int usbh_uvc_preview_ai_is_busy(void)
{
    return uvc_ai_processing_busy ? 1 : 0;
}

int usbh_uvc_preview_start(uint8_t fmt, uint16_t w, uint16_t h)
{
    uint32_t frame_bufsize = UVC_FRAME_BUF_SIZE;
    int ret;

    if (!uvc_device_ready) {
        USB_LOG_WRN("UVC camera is not ready\r\n");
        return -RT_ERROR;
    }

    if (uvc_app_running) {
        USB_LOG_WRN("UVC already running, stop first\r\n");
        return -RT_EBUSY;
    }

    uvc_ai_app_enforce_mode(&fmt, &w, &h);

    USB_LOG_INFO("UVC start: %ux%u format=%s\r\n", w, h,
                 fmt == USBH_VIDEO_FORMAT_MJPEG ? "mjpeg" : "yuyv");

    if (fmt == USBH_VIDEO_FORMAT_UNCOMPRESSED) {
        uint32_t yuyv_size = (uint32_t)w * (uint32_t)h * 2U;

        if ((w == 0U) || (h == 0U) || (yuyv_size > UVC_FRAME_BUF_SIZE)) {
            USB_LOG_ERR("Invalid YUYV frame size: %ux%u (%lu)\r\n",
                        w, h, (unsigned long)yuyv_size);
            return -RT_EINVAL;
        }
        frame_bufsize = yuyv_size;
    }

    /* initialise frame pool */
    {
        for (uint32_t i = 0; i < UVC_FRAME_BUF_COUNT; i++) {
            frame_pool[i].frame_buf = frame_buffer[i];
            frame_pool[i].frame_bufsize = frame_bufsize;
        }
    }

    uvc_cam_w = w;
    uvc_cam_h = h;
    g_uvc_display_fps = 0;
    g_uvc_display_drop_count = 0;
    uvc_last_capture_ok = 0U;
    uvc_display_freeze = 0U;
    uvc_ai_process_request = 0U;
    uvc_ai_processing_busy = 0U;
    uvc_force_live_redraw = 0U;

    ret = usbh_video_stream_create(frame_pool, UVC_FRAME_BUF_COUNT);
    if (ret < 0) {
        USB_LOG_ERR("UVC frame pool create failed: %d\r\n", ret);
        return ret;
    }

    uvc_app_running = 1;

    /* Spawn a persistent frame worker before starting the stream. */
    rt_thread_t t;

    t = rt_thread_create("uvc_frm", uvc_frame_thread_entry, NULL,
                         65536, 22, 10);
    if (t) {
        rt_thread_startup(t);
    } else {
        USB_LOG_ERR("UVC preview thread create failed\r\n");
        uvc_app_running = 0U;
        return -RT_ENOMEM;
    }

    rt_thread_mdelay(50);

    usbh_video_stream_start(w, h, fmt);

    return 0;
}

/* ---------- msh commands ---------- */

static int cmd_usbh_uvc_start(int argc, char **argv)
{
    uint8_t fmt = USBH_VIDEO_FORMAT_UNCOMPRESSED;  /* default: YUYV */
    uint16_t w = 320;
    uint16_t h = 240;

    if (argc >= 2) {
        fmt = atoi(argv[1]);  /* 0 = YUYV, 1 = MJPEG */
    }
    if (argc >= 4) {
        w = atoi(argv[2]);
        h = atoi(argv[3]);
    }

    return usbh_uvc_preview_start(fmt, w, h);
}
MSH_CMD_EXPORT_ALIAS(cmd_usbh_uvc_start, usbh_uvc_start,
                     Start UVC stream: usbh_uvc_start [fmt] [w] [h]);

static int cmd_usbh_uvc_stop(int argc, char **argv)
{
    uvc_app_running = 0;
    rt_thread_mdelay(100);  /* let helper threads notice the flag and exit */
    usbh_video_stream_stop();
    USB_LOG_INFO("UVC stopped\r\n");
    return 0;
}
MSH_CMD_EXPORT_ALIAS(cmd_usbh_uvc_stop, usbh_uvc_stop, Stop UVC stream);
