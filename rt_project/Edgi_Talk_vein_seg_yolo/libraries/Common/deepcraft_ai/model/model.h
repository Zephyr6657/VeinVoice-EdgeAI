#ifndef VEIN_MODEL_H
#define VEIN_MODEL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define IMAI_MODEL_SIZE 121168U
#define IMAI_DATAIN_COUNT (160U * 160U * 3U)
#define IMAI_DATAOUT_COUNT (160U * 160U)
#define IMAI_RET_SUCCESS 0
#define IMAI_RET_NODATA -1
#define IMAI_RET_ERROR -2
#define IMAI_RET_STREAMEND -3

void IMAI_compute(const int8_t *restrict datain, int8_t *restrict dataout);
void IMAI_finalize(void);
int IMAI_init(void);

#ifdef __cplusplus
}
#endif

#endif /* VEIN_MODEL_H */
