/*******************************************************************************
#include <packages/lvgl_9.2.0/src/draw/sw/lv_draw_sw.h>
* File Name        : lv_port_disp.c
*
* Description      : This file provides implementation of low level display
*                    device driver for LVGL.
*
* Related Document : See README.md
*
******************************************************************************/

/*******************************************************************************
* Header Files
*******************************************************************************/
#include "lv_port_disp.h"
#include <stdbool.h>
#include <string.h>
#include <rtdevice.h>
#include "cy_graphics.h"
#include "usbh_uvc_display_hook.h"


/*******************************************************************************
* Global Variables
*******************************************************************************/
CY_SECTION(".cy_gpu_buf") LV_ATTRIBUTE_MEM_ALIGN uint8_t disp_buf1[MY_DISP_HOR_RES *
                                               MY_DISP_VER_RES * 2];
CY_SECTION(".cy_gpu_buf") LV_ATTRIBUTE_MEM_ALIGN uint8_t disp_buf2[MY_DISP_HOR_RES *
                                               MY_DISP_VER_RES * 2];
/* Frame buffers used by GFXSS to render UI */
void *frame_buffer1 = &disp_buf1;
void *frame_buffer2 = &disp_buf2;

cy_stc_gfx_context_t gfx_context;
extern uint8_t graphics_buffer[];

static void copy_lvgl_frame_preserve_uvc_viewport(uint8_t *dst, const uint8_t *src)
{
    uint16_t vx;
    uint16_t vy;
    uint16_t vw;
    uint16_t vh;
    const uint32_t row_bytes = MY_DISP_HOR_RES * 2U;

    if (!uvc_display_get_viewport(&vx, &vy, &vw, &vh)) {
        memcpy(dst, src, MY_DISP_HOR_RES * MY_DISP_VER_RES * 2U);
        return;
    }

    if ((vx >= MY_DISP_HOR_RES) || (vy >= MY_DISP_VER_RES) ||
        (vw == 0U) || (vh == 0U)) {
        memcpy(dst, src, MY_DISP_HOR_RES * MY_DISP_VER_RES * 2U);
        return;
    }

    if ((uint32_t)vx + vw > MY_DISP_HOR_RES) {
        vw = (uint16_t)(MY_DISP_HOR_RES - vx);
    }
    if ((uint32_t)vy + vh > MY_DISP_VER_RES) {
        vh = (uint16_t)(MY_DISP_VER_RES - vy);
    }

    for (uint16_t y = 0U; y < MY_DISP_VER_RES; y++) {
        uint8_t *dst_row = dst + (uint32_t)y * row_bytes;
        const uint8_t *src_row = src + (uint32_t)y * row_bytes;

        if ((y < vy) || (y >= (uint16_t)(vy + vh))) {
            memcpy(dst_row, src_row, row_bytes);
            continue;
        }

        if (vx > 0U) {
            memcpy(dst_row, src_row, (uint32_t)vx * 2U);
        }

        if ((uint32_t)vx + vw < MY_DISP_HOR_RES) {
            uint32_t right_x = (uint32_t)vx + vw;
            memcpy(dst_row + right_x * 2U,
                   src_row + right_x * 2U,
                   (MY_DISP_HOR_RES - right_x) * 2U);
        }
    }
}


/*******************************************************************************
* Function Name: disp_flush
********************************************************************************
* Summary:
*  Flush the content of the internal buffer the specific area on the display.
*  You can use DMA or any hardware acceleration to do this operation in the
*  background but 'lv_disp_flush_ready()' has to be called when finished.
*
* Parameters:
*  *disp_drv: Pointer to the display driver structure to be registered by HAL.
*  *area: Pointer to the area of the screen (not used).
*  *color_p: Pointer to the frame buffer address.
*
* Return:
*  void
*
*******************************************************************************/
static void LV_ATTRIBUTE_FAST_MEM disp_flush(lv_display_t *disp_drv, const lv_area_t *area,
        uint8_t *color_p)
{
    CY_UNUSED_PARAMETER(area);
    rt_device_t lcd;

    copy_lvgl_frame_preserve_uvc_viewport(graphics_buffer, color_p);

    lcd = rt_device_find("lcd");
    if (lcd != RT_NULL)
    {
        rt_device_control(lcd, RTGRAPHIC_CTRL_RECT_UPDATE, RT_NULL);
    }
    else
    {
        Cy_GFXSS_Set_FrameBuffer((GFXSS_Type*) GFXSS, (uint32_t*) graphics_buffer,
                                 &gfx_context);
    }

    /* Inform the graphics library that you are ready with the flushing */
    lv_display_flush_ready(disp_drv);

}


/*******************************************************************************
* Function Name: lv_port_disp_init
********************************************************************************
* Summary:
*  Initialization function for display devices supported by LittelvGL.
*   LVGL requires a buffer where it internally draws the widgets.
*   Later this buffer will passed to your display driver's `flush_cb` to copy
*   its content to your display.
*   The buffer has to be greater than 1 display row
*
*   There are 3 buffering configurations:
*   1. Create ONE buffer:
*      LVGL will draw the display's content here and writes it to your display
*
*   2. Create TWO buffer:
*      LVGL will draw the display's content to a buffer and writes it your
*      display.
*      You should use DMA to write the buffer's content to the display.
*      It will enable LVGL to draw the next part of the screen to the other
*      buffer while the data is being sent form the first buffer.
*      It makes rendering and flushing parallel.
*
*   3. Double buffering
*      Set 2 screens sized buffers and set disp_drv.full_refresh = 1.
*      This way LVGL will always provide the whole rendered screen in `flush_cb`
*      and you only need to change the frame buffer's address.
*
*
* Parameters:
*  void
*
* Return:
*  void
*
*******************************************************************************/
void lv_port_disp_init(void)
{
    memset(disp_buf1, 0, sizeof(disp_buf1));
    memset(disp_buf2, 0, sizeof(disp_buf2));

    lv_display_t *disp = lv_display_create(MY_DISP_HOR_RES, MY_DISP_VER_RES);

    lv_display_set_flush_cb(disp, disp_flush);

    lv_tick_set_cb(&rt_tick_get_millisecond);

    lv_display_set_buffers(disp, disp_buf1, disp_buf2, sizeof(disp_buf1),
                           LV_DISPLAY_RENDER_MODE_FULL);//

    // lv_display_set_rotation(disp, LV_DISPLAY_ROTATION_270);

    Cy_GFXSS_Clear_DC_Interrupt((GFXSS_Type*) GFXSS, &gfx_context);
}



/* [] END OF FILE */
