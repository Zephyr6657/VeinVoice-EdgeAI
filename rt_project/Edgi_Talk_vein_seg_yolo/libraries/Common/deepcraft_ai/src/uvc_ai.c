#include <rtthread.h>
#include <string.h>

#include "uvc_ai.h"
#include "model.h"
#include "mtb_ml.h"
#include "usb_config.h"

#undef USB_DBG_TAG
#define USB_DBG_TAG "uvc_ai"
#include "usb_log.h"

#define UVC_AI_ITCM_SECTION __attribute__((section(".cy_itcm")))
#define UVC_AI_DTCM_SECTION __attribute__((section(".cy_dtcm")))
#define UVC_AI_ML_INPUT_SECTION __attribute__((section(".m33_m55_shared_hyperram")))
#define UVC_AI_ALIGN_16 __attribute__((aligned(16)))
#define UVC_AI_SRC_STRIDE_BYTES (UVC_AI_CAMERA_WIDTH * 2U)
#define UVC_AI_FRAME_BYTES (UVC_AI_SRC_STRIDE_BYTES * UVC_AI_CAMERA_HEIGHT)
#define UVC_AI_INPUT_ZERO_POINT (-128)
#define UVC_AI_THRESHOLD_Q ((int8_t)0)  /* sigmoid(0.60) quantized with output scale=0.055185, zp=-7 */

static UVC_AI_ML_INPUT_SECTION UVC_AI_ALIGN_16 int8_t g_model_input[IMAI_DATAIN_COUNT];
static UVC_AI_DTCM_SECTION UVC_AI_ALIGN_16 int8_t g_model_output[IMAI_DATAOUT_COUNT];

static uint8_t g_ai_initialized;
static uint16_t g_src_width;
static uint16_t g_src_height;
static uint8_t g_uvc_ai_yuv_lut_ready;
static rt_tick_t g_last_ai_log_tick;
static UVC_AI_DTCM_SECTION int32_t g_uvc_ai_y_lut[256];
static UVC_AI_DTCM_SECTION int32_t g_uvc_ai_u_to_b_lut[256];
static UVC_AI_DTCM_SECTION int32_t g_uvc_ai_u_to_g_lut[256];
static UVC_AI_DTCM_SECTION int32_t g_uvc_ai_v_to_r_lut[256];
static UVC_AI_DTCM_SECTION int32_t g_uvc_ai_v_to_g_lut[256];

static inline uint8_t uvc_ai_sat_to_u8(int32_t value)
{
#if defined(__ARM_FEATURE_DSP) || defined(__ARM_FEATURE_MVE)
    return (uint8_t)__USAT(value, 8U);
#else
    if (value < 0) {
        value = 0;
    }
    if (value > 255) {
        value = 255;
    }
    return (uint8_t)value;
#endif
}

static void uvc_ai_prepare_yuv_lut(void)
{
    uint32_t i;

    if (g_uvc_ai_yuv_lut_ready) {
        return;
    }

    for (i = 0U; i < 256U; i++) {
        int32_t y = (int32_t)i - 16;
        int32_t uv = (int32_t)i - 128;

        if (y < 0) {
            y = 0;
        }

        g_uvc_ai_y_lut[i] = 298 * y;
        g_uvc_ai_u_to_b_lut[i] = 516 * uv;
        g_uvc_ai_u_to_g_lut[i] = -100 * uv;
        g_uvc_ai_v_to_r_lut[i] = 409 * uv;
        g_uvc_ai_v_to_g_lut[i] = -208 * uv;
    }

    g_uvc_ai_yuv_lut_ready = 1U;
}

static UVC_AI_ITCM_SECTION void uvc_ai_yuyv_to_model_input_160(const uint8_t *restrict yuyv)
{
    uint32_t dy;
    int8_t *restrict dst = g_model_input;
    const int32_t *restrict y_lut = g_uvc_ai_y_lut;
    const int32_t *restrict u_to_b_lut = g_uvc_ai_u_to_b_lut;
    const int32_t *restrict u_to_g_lut = g_uvc_ai_u_to_g_lut;
    const int32_t *restrict v_to_r_lut = g_uvc_ai_v_to_r_lut;
    const int32_t *restrict v_to_g_lut = g_uvc_ai_v_to_g_lut;

    for (dy = 0U; dy < UVC_AI_MODEL_HEIGHT; dy++) {
        uint32_t sy = (dy * UVC_AI_CAMERA_HEIGHT) / UVC_AI_MODEL_HEIGHT;
        const uint8_t *srow = yuyv + sy * UVC_AI_SRC_STRIDE_BYTES;
        uint32_t dx;

        for (dx = 0U; dx < UVC_AI_MODEL_WIDTH; dx++) {
            uint32_t sx = (dx * UVC_AI_CAMERA_WIDTH) / UVC_AI_MODEL_WIDTH;
            const uint8_t *pair = srow + (sx & ~1U) * 2U;
            uint8_t yv = pair[(sx & 1U) ? 2U : 0U];
            uint8_t u = pair[1U];
            uint8_t v = pair[3U];
            int32_t y_term = y_lut[yv];
            int32_t uv_to_g = u_to_g_lut[u] + v_to_g_lut[v];
            uint8_t r = uvc_ai_sat_to_u8((y_term + v_to_r_lut[v] + 128) >> 8);
            uint8_t g = uvc_ai_sat_to_u8((y_term + uv_to_g + 128) >> 8);
            uint8_t b = uvc_ai_sat_to_u8((y_term + u_to_b_lut[u] + 128) >> 8);

            *dst++ = (int8_t)((int32_t)r + UVC_AI_INPUT_ZERO_POINT);
            *dst++ = (int8_t)((int32_t)g + UVC_AI_INPUT_ZERO_POINT);
            *dst++ = (int8_t)((int32_t)b + UVC_AI_INPUT_ZERO_POINT);
        }
    }
}

void uvc_ai_reset_result(uvc_ai_result_t *result)
{
    if (result != RT_NULL) {
        memset(result, 0, sizeof(*result));
    }
}

int uvc_ai_init(const uvc_ai_config_t *config)
{
    int ret;

    if (config != RT_NULL) {
        g_src_width = (config->src_width == 0U) ? UVC_AI_CAMERA_WIDTH : config->src_width;
        g_src_height = (config->src_height == 0U) ? UVC_AI_CAMERA_HEIGHT : config->src_height;
    } else {
        g_src_width = UVC_AI_CAMERA_WIDTH;
        g_src_height = UVC_AI_CAMERA_HEIGHT;
    }

    if ((g_src_width != UVC_AI_CAMERA_WIDTH) || (g_src_height != UVC_AI_CAMERA_HEIGHT)) {
        USB_LOG_WRN("AI only supports 320x240 UVC input, force %ux%u\r\n",
                    (unsigned)UVC_AI_CAMERA_WIDTH, (unsigned)UVC_AI_CAMERA_HEIGHT);
        g_src_width = UVC_AI_CAMERA_WIDTH;
        g_src_height = UVC_AI_CAMERA_HEIGHT;
    }

    uvc_ai_prepare_yuv_lut();

    if (g_ai_initialized) {
        return RT_EOK;
    }

    mtb_ml_set_cache_mgmt_type(MTB_ML_ETHOSU_CACHE_MGMT_OUTER_LAYERS);

    ret = IMAI_init();
    if (ret != IMAI_RET_SUCCESS) {
        USB_LOG_ERR("IMAI_init failed: %d\r\n", ret);
        return -RT_ERROR;
    }

    g_ai_initialized = 1U;
    USB_LOG_INFO("Vein AI model initialized input=160x160 threshold_q=%d\r\n",
                 (int)UVC_AI_THRESHOLD_Q);
    return RT_EOK;
}

void uvc_ai_deinit(void)
{
    if (!g_ai_initialized) {
        return;
    }

    IMAI_finalize();
    g_ai_initialized = 0U;
    USB_LOG_INFO("Vein AI model deinitialized\r\n");
}

UVC_AI_ITCM_SECTION int uvc_ai_process_yuyv(const uint8_t *yuyv, uint32_t yuyv_size, uvc_ai_result_t *result)
{
    rt_tick_t prep_start_tick;
    rt_tick_t start_tick;
    rt_tick_t end_tick;
    rt_tick_t log_now_tick;
    uint32_t active = 0U;
    int8_t min_q = 127;
    int8_t max_q = -128;
    uint32_t i;

    if ((!g_ai_initialized) || (yuyv == RT_NULL) || (result == RT_NULL)) {
        return -RT_EINVAL;
    }

    if (yuyv_size < UVC_AI_FRAME_BYTES) {
        USB_LOG_WRN("AI frame too small: %lu < %lu\r\n",
                    (unsigned long)yuyv_size, (unsigned long)UVC_AI_FRAME_BYTES);
        return -RT_EINVAL;
    }

    prep_start_tick = rt_tick_get();
    uvc_ai_yuyv_to_model_input_160(yuyv);

    start_tick = rt_tick_get();
    IMAI_compute(g_model_input, g_model_output);
    end_tick = rt_tick_get();

    for (i = 0U; i < UVC_AI_MASK_PIXELS; i++) {
        if (g_model_output[i] < min_q) {
            min_q = g_model_output[i];
        }
        if (g_model_output[i] > max_q) {
            max_q = g_model_output[i];
        }
        if (g_model_output[i] >= UVC_AI_THRESHOLD_Q) {
            active++;
        }
    }

    uvc_ai_reset_result(result);
    result->valid = 1U;
    result->mask_q = g_model_output;
    result->threshold_q = (uint8_t)UVC_AI_THRESHOLD_Q;
    result->active_pixels = active;
    result->inference_ms = ((float)(end_tick - start_tick) * 1000.0f) / (float)RT_TICK_PER_SECOND;

    log_now_tick = rt_tick_get();
    if ((g_last_ai_log_tick == 0U) ||
        ((log_now_tick - g_last_ai_log_tick) >= (3U * RT_TICK_PER_SECOND))) {
        uint32_t inference_ms = ((uint32_t)(end_tick - start_tick) * 1000U) / RT_TICK_PER_SECOND;
        uint32_t prep_ms = ((uint32_t)(start_tick - prep_start_tick) * 1000U) / RT_TICK_PER_SECOND;
        USB_LOG_INFO("Vein AI inference %lu ms prep %lu ms active=%lu/%u q=[%d,%d]\r\n",
                     (unsigned long)inference_ms,
                     (unsigned long)prep_ms,
                     (unsigned long)active, (unsigned)UVC_AI_MASK_PIXELS,
                     (int)min_q, (int)max_q);
        g_last_ai_log_tick = log_now_tick;
    }

    return RT_EOK;
}

const int8_t *uvc_ai_get_latest_mask(void)
{
    return g_model_output;
}

uint8_t uvc_ai_get_threshold_q(void)
{
    return (uint8_t)UVC_AI_THRESHOLD_Q;
}
