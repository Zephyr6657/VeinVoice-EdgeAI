#ifndef EDGETALK_UI_H
#define EDGETALK_UI_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void edgetalk_ui_init(void);
int edgetalk_ui_main_page_ready(void);
void edgetalk_ui_set_ai_report(const char *text);
void edgetalk_ui_post_ai_reply(const char *text);
void edgetalk_ui_post_ai_error(const char *text);
void edgetalk_ui_voice_result_ready(const char *text);
void edgetalk_ui_voice_error(const char *text);
void edgetalk_ui_set_circulation_score(uint8_t score);
void edgetalk_ui_clear_circulation_score(void);

int edgetalk_voice_start_request(void);
int edgetalk_voice_stop_request(void);

#ifdef __cplusplus
}
#endif

#endif /* EDGETALK_UI_H */
