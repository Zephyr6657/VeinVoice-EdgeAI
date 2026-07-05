#include <rtthread.h>
#include <rtdevice.h>
#include <board.h>
#include <lv_rt_thread_conf.h>
#include "usbh_core.h"
#include "edgetalk_ui.h"

#define LED_PIN_G               GET_PIN(16, 6)
#define CAMERA_TRIGGER_PIN      GET_PIN(8, 3)  /* SW2 / CYBSP_USER_BTN, active low */
#define CAMERA_PREVIEW_FMT      0U      /* 0 = YUYV, 1 = MJPEG */
#define CAMERA_PREVIEW_WIDTH    320U
#define CAMERA_PREVIEW_HEIGHT   240U

extern int usbh_uvc_preview_is_ready(void);
extern int usbh_uvc_preview_is_running(void);
extern int usbh_uvc_preview_start(uint8_t fmt, uint16_t w, uint16_t h);
extern int usbh_uvc_preview_toggle_freeze(void);

static void camera_preview_auto_thread(void *parameter)
{
    rt_base_t last_state;
    rt_base_t now_state;

    (void)parameter;

    rt_kprintf("Camera preview: waiting for UVC camera...\r\n");

    while (!usbh_uvc_preview_is_ready())
    {
        rt_thread_mdelay(500);
    }

    rt_pin_mode(CAMERA_TRIGGER_PIN, PIN_MODE_INPUT_PULLUP);
    last_state = rt_pin_read(CAMERA_TRIGGER_PIN);

    rt_kprintf("Camera preview: camera ready, starting continuous %ux%u YUYV stream\r\n",
               CAMERA_PREVIEW_WIDTH, CAMERA_PREVIEW_HEIGHT);

    if (usbh_uvc_preview_start(CAMERA_PREVIEW_FMT,
                               CAMERA_PREVIEW_WIDTH,
                               CAMERA_PREVIEW_HEIGHT) != 0)
    {
        rt_kprintf("Camera preview: auto start failed, use msh: usbh_uvc_start 0 320 240\r\n");
    }
    else
    {
        rt_kprintf("Camera preview: press SW2/User Button to freeze/resume the current frame\r\n");
    }

    while (1)
    {
        now_state = rt_pin_read(CAMERA_TRIGGER_PIN);
        if ((last_state != PIN_LOW) && (now_state == PIN_LOW))
        {
            rt_thread_mdelay(30);
            if (rt_pin_read(CAMERA_TRIGGER_PIN) == PIN_LOW)
            {
                if (usbh_uvc_preview_is_running())
                {
                    int frozen = usbh_uvc_preview_toggle_freeze();
                    rt_kprintf("Camera preview: %s\r\n", frozen ? "frame frozen" : "live resumed");
                }
                else
                {
                    rt_kprintf("Camera preview: stream is not running, restart requested\r\n");
                    (void)usbh_uvc_preview_start(CAMERA_PREVIEW_FMT,
                                                 CAMERA_PREVIEW_WIDTH,
                                                 CAMERA_PREVIEW_HEIGHT);
                }

                while (rt_pin_read(CAMERA_TRIGGER_PIN) == PIN_LOW)
                {
                    rt_thread_mdelay(20);
                }
                last_state = PIN_HIGH;
                rt_thread_mdelay(300);
            }
        }

        last_state = now_state;
        rt_thread_mdelay(20);
    }
}

static void camera_preview_auto_start(void)
{
    rt_thread_t tid;

    tid = rt_thread_create("cam_auto",
                           camera_preview_auto_thread,
                           RT_NULL,
                           2048,
                           24,
                           10);
    if (tid != RT_NULL)
    {
        rt_thread_startup(tid);
    }
    else
    {
        rt_kprintf("Camera preview: failed to create auto-start thread\r\n");
    }
}

void lv_user_gui_init(void)
{
    edgetalk_ui_init();
}

int main(void)
{
    rt_kprintf("Hello RT-Thread\r\n");
    rt_kprintf("It's cortex-m55\r\n");

    lvgl_thread_init();

    /* USB host initialize (same as msh: usbh_init 0 0x44900000) */
    usbh_initialize(0, USBHS_BASE, NULL);
    camera_preview_auto_start();

    rt_pin_mode(LED_PIN_G, PIN_MODE_OUTPUT);

    while (1)
    {
        rt_pin_write(LED_PIN_G, PIN_LOW);
        rt_thread_mdelay(500);
        rt_pin_write(LED_PIN_G, PIN_HIGH);
        rt_thread_mdelay(500);
    }
    return 0;
}
