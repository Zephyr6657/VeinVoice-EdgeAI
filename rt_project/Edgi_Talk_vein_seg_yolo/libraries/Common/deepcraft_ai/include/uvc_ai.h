#ifndef UVC_AI_H
#define UVC_AI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define UVC_AI_CAMERA_WIDTH      320U
#define UVC_AI_CAMERA_HEIGHT     240U
#define UVC_AI_MODEL_WIDTH       160U
#define UVC_AI_MODEL_HEIGHT      160U
#define UVC_AI_MODEL_CHANNELS    3U
#define UVC_AI_MASK_PIXELS       (UVC_AI_MODEL_WIDTH * UVC_AI_MODEL_HEIGHT)
#define UVC_AI_FRAMES_TO_SKIP    4U

typedef struct {
    uint8_t initialized;
    uint16_t src_width;
    uint16_t src_height;
} uvc_ai_config_t;

typedef struct {
    uint8_t valid;
    float inference_ms;
    uint32_t active_pixels;
    uint8_t threshold_q;
    const int8_t *mask_q;
} uvc_ai_result_t;

int uvc_ai_init(const uvc_ai_config_t *config);
void uvc_ai_deinit(void);
void uvc_ai_reset_result(uvc_ai_result_t *result);
int uvc_ai_process_yuyv(const uint8_t *yuyv, uint32_t yuyv_size, uvc_ai_result_t *result);
const int8_t *uvc_ai_get_latest_mask(void);
uint8_t uvc_ai_get_threshold_q(void);

#ifdef __cplusplus
}
#endif

#endif /* UVC_AI_H */
