#ifndef UVC_AI_APP_H
#define UVC_AI_APP_H

#include <rtthread.h>
#include <stdint.h>
#include <string.h>

#include "usbh_uvc_stream.h"
#include "usbh_uvc_display_hook.h"
#include "uvc_ai.h"
#include "edgetalk_ui.h"

#ifdef __cplusplus
extern "C" {
#endif

static uvc_ai_result_t g_uvc_ai_result;
static uint32_t g_uvc_ai_frame_index;

#define UVC_AI_DISPLAY_MEM __attribute__((section(".m33_m55_shared_hyperram"), aligned(16)))

static UVC_AI_DISPLAY_MEM uint8_t g_uvc_ai_display_mask[UVC_AI_MASK_PIXELS];
static UVC_AI_DISPLAY_MEM uint8_t g_uvc_ai_mask_tmp_a[UVC_AI_MASK_PIXELS];
static UVC_AI_DISPLAY_MEM uint8_t g_uvc_ai_mask_tmp_b[UVC_AI_MASK_PIXELS];
static UVC_AI_DISPLAY_MEM uint16_t g_uvc_ai_component_stack[UVC_AI_MASK_PIXELS];

#define UVC_AI_DISPLAY_THRESHOLD_OFFSET 24
#define UVC_AI_DISPLAY_MIN_COMPONENT_AREA 80U

typedef struct {
    uint32_t active_pixels;
    uint32_t largest_component;
    uint16_t component_count;
} uvc_ai_mask_stats_t;

static uint8_t uvc_ai_mask_get(const uint8_t *mask, int32_t x, int32_t y)
{
    if ((x < 0) || (y < 0) ||
        (x >= (int32_t)UVC_AI_MODEL_WIDTH) ||
        (y >= (int32_t)UVC_AI_MODEL_HEIGHT)) {
        return 0U;
    }
    return mask[(uint32_t)y * UVC_AI_MODEL_WIDTH + (uint32_t)x] ? 1U : 0U;
}

static void uvc_ai_mask_dilate3(const uint8_t *src, uint8_t *dst)
{
    uint32_t y;

    for (y = 0U; y < UVC_AI_MODEL_HEIGHT; y++) {
        uint32_t x;
        for (x = 0U; x < UVC_AI_MODEL_WIDTH; x++) {
            uint8_t v = 0U;
            int32_t dy;

            for (dy = -1; dy <= 1 && !v; dy++) {
                int32_t dx;
                for (dx = -1; dx <= 1; dx++) {
                    if (uvc_ai_mask_get(src, (int32_t)x + dx, (int32_t)y + dy)) {
                        v = 255U;
                        break;
                    }
                }
            }
            dst[y * UVC_AI_MODEL_WIDTH + x] = v;
        }
    }
}

static void uvc_ai_mask_erode3(const uint8_t *src, uint8_t *dst)
{
    uint32_t y;

    for (y = 0U; y < UVC_AI_MODEL_HEIGHT; y++) {
        uint32_t x;
        for (x = 0U; x < UVC_AI_MODEL_WIDTH; x++) {
            uint8_t v = 255U;
            int32_t dy;

            for (dy = -1; dy <= 1 && v; dy++) {
                int32_t dx;
                for (dx = -1; dx <= 1; dx++) {
                    if (!uvc_ai_mask_get(src, (int32_t)x + dx, (int32_t)y + dy)) {
                        v = 0U;
                        break;
                    }
                }
            }
            dst[y * UVC_AI_MODEL_WIDTH + x] = v;
        }
    }
}

static void uvc_ai_mask_remove_isolated(const uint8_t *src, uint8_t *dst)
{
    uint32_t y;

    for (y = 0U; y < UVC_AI_MODEL_HEIGHT; y++) {
        uint32_t x;
        for (x = 0U; x < UVC_AI_MODEL_WIDTH; x++) {
            uint8_t count = 0U;
            int32_t dy;

            for (dy = -1; dy <= 1; dy++) {
                int32_t dx;
                for (dx = -1; dx <= 1; dx++) {
                    count = (uint8_t)(count + uvc_ai_mask_get(src, (int32_t)x + dx, (int32_t)y + dy));
                }
            }
            dst[y * UVC_AI_MODEL_WIDTH + x] = (count >= 3U) ? src[y * UVC_AI_MODEL_WIDTH + x] : 0U;
        }
    }
}

static void uvc_ai_mask_filter_components(uint8_t *work, uint8_t *dst, uint16_t min_area)
{
    uint32_t i;

    memset(dst, 0, UVC_AI_MASK_PIXELS);

    for (i = 0U; i < UVC_AI_MASK_PIXELS; i++) {
        uint32_t count = 0U;
        uint32_t sp = 0U;
        uint32_t read = 0U;

        if (work[i] == 0U) {
            continue;
        }

        work[i] = 128U;
        g_uvc_ai_component_stack[sp++] = (uint16_t)i;

        while (read < sp) {
            uint16_t index = g_uvc_ai_component_stack[read++];
            int32_t x = (int32_t)(index % UVC_AI_MODEL_WIDTH);
            int32_t y = (int32_t)(index / UVC_AI_MODEL_WIDTH);
            int32_t dy;

            count++;
            for (dy = -1; dy <= 1; dy++) {
                int32_t dx;
                for (dx = -1; dx <= 1; dx++) {
                    int32_t nx;
                    int32_t ny;
                    uint32_t nindex;

                    if ((dx == 0) && (dy == 0)) {
                        continue;
                    }

                    nx = x + dx;
                    ny = y + dy;
                    if ((nx < 0) || (ny < 0) ||
                        (nx >= (int32_t)UVC_AI_MODEL_WIDTH) ||
                        (ny >= (int32_t)UVC_AI_MODEL_HEIGHT)) {
                        continue;
                    }

                    nindex = (uint32_t)ny * UVC_AI_MODEL_WIDTH + (uint32_t)nx;
                    if (work[nindex] != 0U && work[nindex] != 128U) {
                        work[nindex] = 128U;
                        if (sp < UVC_AI_MASK_PIXELS) {
                            g_uvc_ai_component_stack[sp++] = (uint16_t)nindex;
                        }
                    }
                }
            }
        }

        if (count >= (uint32_t)min_area) {
            uint32_t j;
            for (j = 0U; j < sp; j++) {
                dst[g_uvc_ai_component_stack[j]] = 255U;
            }
        }
    }
}

static void uvc_ai_collect_mask_stats(const uint8_t *src, uint8_t *work, uvc_ai_mask_stats_t *stats)
{
    uint32_t i;

    memset(stats, 0, sizeof(*stats));
    memcpy(work, src, UVC_AI_MASK_PIXELS);

    for (i = 0U; i < UVC_AI_MASK_PIXELS; i++) {
        uint32_t count = 0U;
        uint32_t sp = 0U;
        uint32_t read = 0U;

        if (work[i] == 0U) {
            continue;
        }

        work[i] = 128U;
        g_uvc_ai_component_stack[sp++] = (uint16_t)i;

        while (read < sp) {
            uint16_t index = g_uvc_ai_component_stack[read++];
            int32_t x = (int32_t)(index % UVC_AI_MODEL_WIDTH);
            int32_t y = (int32_t)(index / UVC_AI_MODEL_WIDTH);
            int32_t dy;

            count++;
            for (dy = -1; dy <= 1; dy++) {
                int32_t dx;
                for (dx = -1; dx <= 1; dx++) {
                    int32_t nx;
                    int32_t ny;
                    uint32_t nindex;

                    if ((dx == 0) && (dy == 0)) {
                        continue;
                    }

                    nx = x + dx;
                    ny = y + dy;
                    if ((nx < 0) || (ny < 0) ||
                        (nx >= (int32_t)UVC_AI_MODEL_WIDTH) ||
                        (ny >= (int32_t)UVC_AI_MODEL_HEIGHT)) {
                        continue;
                    }

                    nindex = (uint32_t)ny * UVC_AI_MODEL_WIDTH + (uint32_t)nx;
                    if (work[nindex] != 0U && work[nindex] != 128U) {
                        work[nindex] = 128U;
                        if (sp < UVC_AI_MASK_PIXELS) {
                            g_uvc_ai_component_stack[sp++] = (uint16_t)nindex;
                        }
                    }
                }
            }
        }

        stats->active_pixels += count;
        stats->component_count++;
        if (count > stats->largest_component) {
            stats->largest_component = count;
        }
    }
}

static uint8_t uvc_ai_score_from_stats(const uvc_ai_mask_stats_t *stats)
{
    uint32_t area_permille;
    uint32_t visibility;
    uint32_t continuity;
    uint32_t stability;
    uint32_t score;

    if ((stats == RT_NULL) || (stats->active_pixels == 0U)) {
        return 0U;
    }

    area_permille = (stats->active_pixels * 1000U) / UVC_AI_MASK_PIXELS;

    if (area_permille < 5U) {
        visibility = (area_permille * 30U) / 5U;
    } else if (area_permille < 20U) {
        visibility = 30U + ((area_permille - 5U) * 50U) / 15U;
    } else if (area_permille <= 180U) {
        visibility = 80U + ((area_permille - 20U) * 20U) / 160U;
    } else if (area_permille <= 350U) {
        visibility = 100U - ((area_permille - 180U) * 60U) / 170U;
    } else {
        visibility = 20U;
    }

    continuity = (stats->largest_component * 100U) / stats->active_pixels;
    if (continuity > 100U) {
        continuity = 100U;
    }

    if (stats->component_count <= 2U) {
        stability = 100U;
    } else if (stats->component_count <= 8U) {
        stability = 100U - ((uint32_t)(stats->component_count - 2U) * 7U);
    } else if (stats->component_count <= 20U) {
        stability = 58U - ((uint32_t)(stats->component_count - 8U) * 3U);
    } else {
        stability = 20U;
    }

    score = (visibility * 40U + continuity * 40U + stability * 20U + 50U) / 100U;
    if (score > 100U) {
        score = 100U;
    }
    return (uint8_t)score;
}

static uint8_t uvc_ai_app_update_circulation_score(void)
{
    uvc_ai_mask_stats_t stats;
    uint8_t score;
    uint32_t area_permille;

    uvc_ai_collect_mask_stats(g_uvc_ai_display_mask, g_uvc_ai_mask_tmp_a, &stats);
    score = uvc_ai_score_from_stats(&stats);
    area_permille = (stats.active_pixels * 1000U) / UVC_AI_MASK_PIXELS;

    edgetalk_ui_set_circulation_score(score);
    rt_kprintf("[I/uvc_ai] circulation_score=%u active_pixels=%lu area_permille=%lu component_count=%u largest_component=%lu\r\n",
               (unsigned)score,
               (unsigned long)stats.active_pixels,
               (unsigned long)area_permille,
               (unsigned)stats.component_count,
               (unsigned long)stats.largest_component);

    return score;
}

static void uvc_ai_app_build_display_mask(void)
{
    const int8_t *mask = g_uvc_ai_result.mask_q;
    int32_t threshold = (int32_t)(int8_t)g_uvc_ai_result.threshold_q + UVC_AI_DISPLAY_THRESHOLD_OFFSET;
    uint32_t i;

    memset(g_uvc_ai_display_mask, 0, sizeof(g_uvc_ai_display_mask));
    if ((!g_uvc_ai_result.valid) || (mask == RT_NULL)) {
        return;
    }

    for (i = 0U; i < UVC_AI_MASK_PIXELS; i++) {
        g_uvc_ai_mask_tmp_a[i] = ((int32_t)mask[i] >= threshold) ? 255U : 0U;
    }

    /* Keep board display conservative. Aggressive bridge/dilate can turn broad
     * false positives into large blocks when the camera domain shifts. */
    uvc_ai_mask_remove_isolated(g_uvc_ai_mask_tmp_a, g_uvc_ai_mask_tmp_b);
    uvc_ai_mask_erode3(g_uvc_ai_mask_tmp_b, g_uvc_ai_mask_tmp_a);
    uvc_ai_mask_dilate3(g_uvc_ai_mask_tmp_a, g_uvc_ai_mask_tmp_b);
    uvc_ai_mask_filter_components(g_uvc_ai_mask_tmp_b,
                                  g_uvc_ai_display_mask,
                                  UVC_AI_DISPLAY_MIN_COMPONENT_AREA);
}

static inline uint16_t uvc_ai_rgb565_blend(uint16_t base, uint8_t r, uint8_t g, uint8_t b, uint8_t alpha)
{
    uint8_t br = (uint8_t)(((base >> 11) & 0x1fU) << 3);
    uint8_t bg = (uint8_t)(((base >> 5) & 0x3fU) << 2);
    uint8_t bb = (uint8_t)((base & 0x1fU) << 3);
    uint8_t nr = (uint8_t)(((uint16_t)br * (255U - alpha) + (uint16_t)r * alpha) / 255U);
    uint8_t ng = (uint8_t)(((uint16_t)bg * (255U - alpha) + (uint16_t)g * alpha) / 255U);
    uint8_t nb = (uint8_t)(((uint16_t)bb * (255U - alpha) + (uint16_t)b * alpha) / 255U);

    return (uint16_t)(((uint16_t)(nr >> 3) << 11) |
                      ((uint16_t)(ng >> 2) << 5) |
                      ((uint16_t)(nb >> 3)));
}

static void uvc_ai_overlay_callback(const uvc_display_overlay_info_t *info, void *user_ctx)
{
    uint16_t y;

    (void)user_ctx;

    if ((info == RT_NULL) || (info->framebuffer == RT_NULL) ||
        (!g_uvc_ai_result.valid) ||
        (info->dst_width == 0U) || (info->dst_height == 0U)) {
        return;
    }

    for (y = 0U; y < info->dst_height; y++) {
        uint32_t my = ((uint32_t)y * UVC_AI_MODEL_HEIGHT) / info->dst_height;
        uint16_t *row = info->framebuffer +
                        ((uint32_t)info->dst_y_offset + y) * info->lcd_width +
                        info->dst_x_offset;
        uint16_t x;

        for (x = 0U; x < info->dst_width; x++) {
            uint32_t mx = ((uint32_t)x * UVC_AI_MODEL_WIDTH) / info->dst_width;
            uint8_t q = g_uvc_ai_display_mask[my * UVC_AI_MODEL_WIDTH + mx];

            if (q != 0U) {
                row[x] = uvc_ai_rgb565_blend(row[x], 32U, 72U, 96U, 72U);
            }
        }
    }
}

static void uvc_ai_app_enforce_mode(uint8_t *fmt, uint16_t *width, uint16_t *height)
{
    if (fmt != RT_NULL) {
        (void)fmt;
    }
    if (width != RT_NULL) {
        *width = UVC_AI_CAMERA_WIDTH;
    }
    if (height != RT_NULL) {
        *height = UVC_AI_CAMERA_HEIGHT;
    }
}

static int uvc_ai_app_start(uint16_t src_width, uint16_t src_height)
{
    uvc_ai_config_t config;

    memset(&g_uvc_ai_result, 0, sizeof(g_uvc_ai_result));
    g_uvc_ai_frame_index = 0U;

    config.initialized = 0U;
    config.src_width = src_width;
    config.src_height = src_height;

    if (uvc_ai_init(&config) != RT_EOK) {
        return -RT_ERROR;
    }

    uvc_display_set_overlay_callback(uvc_ai_overlay_callback, RT_NULL);
    return RT_EOK;
}

static void uvc_ai_app_clear_result(void)
{
    memset(&g_uvc_ai_result, 0, sizeof(g_uvc_ai_result));
    memset(g_uvc_ai_display_mask, 0, sizeof(g_uvc_ai_display_mask));
    g_uvc_ai_frame_index = 0U;
    edgetalk_ui_clear_circulation_score();
}

static int uvc_ai_app_process_frame(const struct usbh_videoframe *frame)
{
    int ret;

    if ((frame == RT_NULL) ||
        (frame->frame_format != USBH_VIDEO_FORMAT_UNCOMPRESSED) ||
        (frame->frame_buf == RT_NULL)) {
        uvc_ai_app_clear_result();
        return -RT_EINVAL;
    }

    g_uvc_ai_frame_index++;
    uvc_ai_reset_result(&g_uvc_ai_result);
    ret = uvc_ai_process_yuyv(frame->frame_buf, frame->frame_size, &g_uvc_ai_result);
    if (ret != RT_EOK) {
        uvc_ai_app_clear_result();
    } else {
        uvc_ai_app_build_display_mask();
        (void)uvc_ai_app_update_circulation_score();
    }

    return ret;
}

static void uvc_ai_app_stop(void)
{
    uvc_display_set_overlay_callback(RT_NULL, RT_NULL);
    memset(&g_uvc_ai_result, 0, sizeof(g_uvc_ai_result));
    memset(g_uvc_ai_display_mask, 0, sizeof(g_uvc_ai_display_mask));
    uvc_ai_deinit();
}

#ifdef __cplusplus
}
#endif

#endif /* UVC_AI_APP_H */
