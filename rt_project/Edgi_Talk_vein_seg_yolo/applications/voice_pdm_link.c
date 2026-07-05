#include <rtconfig.h>

/*
 * PDM support is compiled from libraries/HAL_Drivers/drv_pdm.c when
 * BSP_USING_AUDIO_RECORD is enabled. This file is intentionally kept as a
 * tiny anchor so older generated project files that list voice_pdm_link.c
 * still build without pulling drv_pdm.c in a second time.
 */
