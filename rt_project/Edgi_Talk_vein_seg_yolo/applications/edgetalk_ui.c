#include "edgetalk_ui.h"

#include "lvgl.h"
#include "rtdef.h"
#include "rtthread.h"
#include "usbh_uvc_display_hook.h"
#include "veinsense_start_logo.h"

LV_FONT_DECLARE(edgetalk_font_cn_common_16)

#define UI_FONT_CN_16      (&edgetalk_font_cn_common_16)

#define UI_COLOR_BG        0x0f1720
#define UI_COLOR_PANEL     0x172331
#define UI_COLOR_PANEL_2   0x1e2b3a
#define UI_COLOR_BORDER    0x314256
#define UI_COLOR_TEXT      0xe6edf3
#define UI_COLOR_MUTED     0x91a4b7
#define UI_COLOR_CYAN      0x38c6d9
#define UI_COLOR_GREEN     0x45d483
#define UI_COLOR_YELLOW    0xf0c84b
#define UI_COLOR_RED       0xff6b6b
#define UI_COLOR_DEEP      0x09111a

#define VEIN_PREVIEW_PANEL_X   16
#define VEIN_PREVIEW_PANEL_Y   58
#define VEIN_PREVIEW_FRAME_X   0
#define VEIN_PREVIEW_FRAME_Y   38
#define VEIN_PREVIEW_FRAME_W   436
#define VEIN_PREVIEW_FRAME_H   278
#define VEIN_PREVIEW_PAD       12

#define DEMO_HEART_RATE        76U
#define DEMO_SPO2              98U
#define CIRC_STABLE_MIN        60U
#define CIRC_GOOD_MIN          80U
#define CIRC_DISPLAY_MIN       62U
#define CIRC_DISPLAY_MAX       95U
#define HR_NORMAL_MIN          60U
#define HR_NORMAL_MAX          100U
#define SPO2_NORMAL_MIN        95U

typedef enum
{
    VOICE_UI_IDLE = 0,
    VOICE_UI_RECORDING,
    VOICE_UI_PROCESSING,
    VOICE_UI_RESULT_READY,
    VOICE_UI_ERROR
} voice_ui_state_t;

typedef struct
{
    uint8_t success;
    char text[768];
} ai_report_async_msg_t;

static lv_obj_t *s_voice_button;
static lv_obj_t *s_voice_button_label;
static lv_obj_t *s_report_status_label;
static lv_obj_t *s_report_body_label;
static lv_obj_t *s_hr_value_label;
static lv_obj_t *s_spo2_value_label;
static lv_obj_t *s_circulation_score_label;
static lv_obj_t *s_vitals_button_label;
static voice_ui_state_t s_voice_state = VOICE_UI_IDLE;
static uint8_t s_vitals_visible;
static volatile uint8_t s_main_page_ready;

static uint8_t calibrate_circulation_score(uint8_t raw_score)
{
    uint16_t calibrated;

    if (raw_score > 100U)
    {
        raw_score = 100U;
    }

    calibrated = (uint16_t)CIRC_DISPLAY_MIN +
                 (((uint16_t)raw_score * (uint16_t)(CIRC_DISPLAY_MAX - CIRC_DISPLAY_MIN)) / 100U);

    if (calibrated > CIRC_DISPLAY_MAX)
    {
        calibrated = CIRC_DISPLAY_MAX;
    }

    return (uint8_t)calibrated;
}

rt_weak int edgetalk_voice_start_request(void)
{
    return 0;
}

rt_weak int edgetalk_voice_stop_request(void)
{
    return 0;
}

static void edgetalk_ui_show_main_page(void);
static void edgetalk_ui_show_health_report_page(void);

static lv_obj_t *create_panel(lv_obj_t *parent, int32_t x, int32_t y, int32_t w, int32_t h)
{
    lv_obj_t *panel = lv_obj_create(parent);

    lv_obj_set_pos(panel, x, y);
    lv_obj_set_size(panel, w, h);
    lv_obj_set_style_bg_color(panel, lv_color_hex(UI_COLOR_PANEL), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(panel, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_border_color(panel, lv_color_hex(UI_COLOR_BORDER), LV_PART_MAIN);
    lv_obj_set_style_border_width(panel, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(panel, 8, LV_PART_MAIN);
    lv_obj_set_style_pad_all(panel, 12, LV_PART_MAIN);
    lv_obj_clear_flag(panel, LV_OBJ_FLAG_SCROLLABLE);

    return panel;
}

static lv_obj_t *create_label(lv_obj_t *parent, const char *text, int32_t x, int32_t y,
                              const lv_font_t *font, uint32_t color)
{
    lv_obj_t *label = lv_label_create(parent);

    lv_label_set_text(label, text);
    lv_obj_set_pos(label, x, y);
    lv_obj_set_style_text_color(label, lv_color_hex(color), LV_PART_MAIN);
    lv_obj_set_style_text_font(label, font, LV_PART_MAIN);
    lv_obj_set_style_text_letter_space(label, 0, LV_PART_MAIN);

    return label;
}

static void create_status_pill(lv_obj_t *parent, const char *text, int32_t x, uint32_t color)
{
    lv_obj_t *pill = lv_obj_create(parent);
    lv_obj_t *label;

    lv_obj_set_pos(pill, x, 18);
    lv_obj_set_size(pill, 76, 28);
    lv_obj_set_style_bg_color(pill, lv_color_hex(UI_COLOR_PANEL_2), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(pill, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_border_color(pill, lv_color_hex(color), LV_PART_MAIN);
    lv_obj_set_style_border_width(pill, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(pill, 6, LV_PART_MAIN);
    lv_obj_set_style_pad_all(pill, 0, LV_PART_MAIN);
    lv_obj_clear_flag(pill, LV_OBJ_FLAG_SCROLLABLE);

    label = create_label(pill, text, 0, 0, &lv_font_montserrat_14, color);
    lv_obj_center(label);
}

static lv_obj_t *create_metric_chip(lv_obj_t *parent, const char *title, const char *value,
                                    const char *unit, int32_t x, uint32_t accent)
{
    lv_obj_t *chip = lv_obj_create(parent);
    lv_obj_t *value_label;

    lv_obj_set_pos(chip, x, 326);
    lv_obj_set_size(chip, 132, 36);
    lv_obj_set_style_bg_color(chip, lv_color_hex(UI_COLOR_PANEL_2), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(chip, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_border_color(chip, lv_color_hex(accent), LV_PART_MAIN);
    lv_obj_set_style_border_width(chip, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(chip, 6, LV_PART_MAIN);
    lv_obj_set_style_pad_all(chip, 0, LV_PART_MAIN);
    lv_obj_clear_flag(chip, LV_OBJ_FLAG_SCROLLABLE);

    create_label(chip, title, 8, 5, &lv_font_montserrat_12, UI_COLOR_MUTED);
    value_label = create_label(chip, value, 48, 8, &lv_font_montserrat_16, accent);
    create_label(chip, unit, 82, 10, &lv_font_montserrat_12, UI_COLOR_MUTED);

    return value_label;
}

static void set_report_status(const char *text, uint32_t color)
{
    if (s_report_status_label == NULL)
    {
        return;
    }

    lv_label_set_text(s_report_status_label, text);
    lv_obj_set_style_text_color(s_report_status_label, lv_color_hex(color), LV_PART_MAIN);
}

static void set_voice_state(voice_ui_state_t state)
{
    s_voice_state = state;

    if (s_voice_button_label == NULL)
    {
        return;
    }

    lv_obj_remove_state(s_voice_button, LV_STATE_DISABLED);

    switch (state)
    {
    case VOICE_UI_RECORDING:
        lv_label_set_text(s_voice_button_label, "停止");
        set_report_status("聆听中", UI_COLOR_CYAN);
        edgetalk_ui_set_ai_report("正在聆听，再次点击语音发送。");
        break;
    case VOICE_UI_PROCESSING:
        lv_label_set_text(s_voice_button_label, "等待");
        lv_obj_add_state(s_voice_button, LV_STATE_DISABLED);
        set_report_status("等待", UI_COLOR_YELLOW);
        edgetalk_ui_set_ai_report("等待云端 AI 回复...");
        break;
    case VOICE_UI_RESULT_READY:
        lv_label_set_text(s_voice_button_label, "语音");
        break;
    case VOICE_UI_ERROR:
        lv_label_set_text(s_voice_button_label, "语音");
        break;
    case VOICE_UI_IDLE:
    default:
        lv_label_set_text(s_voice_button_label, "语音");
        break;
    }

    lv_obj_center(s_voice_button_label);
}

static void voice_button_event_cb(lv_event_t *event)
{
    LV_UNUSED(event);

    if (s_voice_state == VOICE_UI_RECORDING)
    {
        if (edgetalk_voice_stop_request() != 0)
        {
            edgetalk_ui_voice_error("停止录音失败。");
            return;
        }

        set_voice_state(VOICE_UI_PROCESSING);
        return;
    }

    if (edgetalk_voice_start_request() != 0)
    {
        edgetalk_ui_voice_error("录音启动失败。");
        return;
    }

    set_voice_state(VOICE_UI_RECORDING);
}

static void vitals_button_event_cb(lv_event_t *event)
{
    LV_UNUSED(event);

    s_vitals_visible = !s_vitals_visible;

    if ((s_hr_value_label == NULL) || (s_spo2_value_label == NULL) ||
        (s_vitals_button_label == NULL))
    {
        return;
    }

    if (s_vitals_visible)
    {
        lv_label_set_text(s_hr_value_label, "76");
        lv_label_set_text(s_spo2_value_label, "98");
        lv_label_set_text(s_vitals_button_label, "隐藏体征");
    }
    else
    {
        lv_label_set_text(s_hr_value_label, "--");
        lv_label_set_text(s_spo2_value_label, "--");
        lv_label_set_text(s_vitals_button_label, "显示体征");
    }
}

static void health_report_button_event_cb(lv_event_t *event)
{
    LV_UNUSED(event);
    edgetalk_ui_show_health_report_page();
}

static void health_report_back_event_cb(lv_event_t *event)
{
    LV_UNUSED(event);
    edgetalk_ui_show_main_page();
}

static void reset_main_page_state(void)
{
    s_voice_button = NULL;
    s_voice_button_label = NULL;
    s_report_status_label = NULL;
    s_report_body_label = NULL;
    s_hr_value_label = NULL;
    s_spo2_value_label = NULL;
    s_circulation_score_label = NULL;
    s_vitals_button_label = NULL;
    s_voice_state = VOICE_UI_IDLE;
    s_vitals_visible = 0;
}

static void create_health_report_button(lv_obj_t *root)
{
    lv_obj_t *button = lv_button_create(root);
    lv_obj_t *label;

    lv_obj_set_pos(button, 334, 18);
    lv_obj_set_size(button, 142, 32);
    lv_obj_set_style_bg_color(button, lv_color_hex(UI_COLOR_PANEL_2), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(button, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_border_color(button, lv_color_hex(UI_COLOR_CYAN), LV_PART_MAIN);
    lv_obj_set_style_border_width(button, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(button, 6, LV_PART_MAIN);
    lv_obj_set_style_pad_all(button, 0, LV_PART_MAIN);
    lv_obj_add_event_cb(button, health_report_button_event_cb, LV_EVENT_CLICKED, NULL);

    label = create_label(button, "健康报告", 0, 0, UI_FONT_CN_16, UI_COLOR_CYAN);
    lv_obj_center(label);
}

static void create_vitals_button(lv_obj_t *root)
{
    lv_obj_t *button = lv_button_create(root);
    lv_obj_t *label;

    lv_obj_set_pos(button, 16, 18);
    lv_obj_set_size(button, 148, 32);
    lv_obj_set_style_bg_color(button, lv_color_hex(UI_COLOR_PANEL_2), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(button, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_border_color(button, lv_color_hex(UI_COLOR_GREEN), LV_PART_MAIN);
    lv_obj_set_style_border_width(button, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(button, 6, LV_PART_MAIN);
    lv_obj_set_style_pad_all(button, 0, LV_PART_MAIN);
    lv_obj_add_event_cb(button, vitals_button_event_cb, LV_EVENT_CLICKED, NULL);

    label = create_label(button, "显示体征", 0, 0, UI_FONT_CN_16, UI_COLOR_GREEN);
    lv_obj_center(label);
    s_vitals_button_label = label;
}

static void create_vein_preview(lv_obj_t *root)
{
    lv_obj_t *panel = create_panel(root, VEIN_PREVIEW_PANEL_X, VEIN_PREVIEW_PANEL_Y, 460, 382);
    lv_obj_t *frame;

    create_label(panel, "静脉预览", 0, 0, UI_FONT_CN_16, UI_COLOR_TEXT);

    frame = lv_obj_create(panel);
    lv_obj_set_pos(frame, VEIN_PREVIEW_FRAME_X, VEIN_PREVIEW_FRAME_Y);
    lv_obj_set_size(frame, VEIN_PREVIEW_FRAME_W, VEIN_PREVIEW_FRAME_H);
    lv_obj_set_style_bg_color(frame, lv_color_hex(0x101922), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(frame, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_border_color(frame, lv_color_hex(0x2c5361), LV_PART_MAIN);
    lv_obj_set_style_border_width(frame, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(frame, 6, LV_PART_MAIN);
    lv_obj_clear_flag(frame, LV_OBJ_FLAG_SCROLLABLE);

    s_hr_value_label = create_metric_chip(panel, "HR", "--", "bpm", 0, UI_COLOR_RED);
    s_spo2_value_label = create_metric_chip(panel, "SpO2", "--", "%", 152, UI_COLOR_CYAN);
    s_circulation_score_label = create_metric_chip(panel, "Circ.", "--", "%", 304, UI_COLOR_GREEN);
}

static void create_report(lv_obj_t *root)
{
    lv_obj_t *panel = create_panel(root, 16, 454, 460, 306);
    lv_obj_t *button;
    lv_obj_t *label;

    create_label(panel, "AI 报告", 0, 0, UI_FONT_CN_16, UI_COLOR_TEXT);
    s_report_status_label = create_label(panel, "等待", 240, 6, UI_FONT_CN_16, UI_COLOR_YELLOW);

    button = lv_button_create(panel);
    lv_obj_set_pos(button, 334, 0);
    lv_obj_set_size(button, 96, 32);
    lv_obj_set_style_bg_color(button, lv_color_hex(UI_COLOR_CYAN), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(button, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_radius(button, 8, LV_PART_MAIN);
    lv_obj_set_style_pad_all(button, 0, LV_PART_MAIN);
    lv_obj_add_event_cb(button, voice_button_event_cb, LV_EVENT_CLICKED, NULL);
    s_voice_button = button;

    label = create_label(button, "语音", 0, 0, UI_FONT_CN_16, 0x071219);
    lv_obj_center(label);
    s_voice_button_label = label;
    lv_obj_move_foreground(button);

    s_report_body_label = create_label(panel,
                                       "点击语音按钮向云端 AI 提问",
                                       0, 44, UI_FONT_CN_16, UI_COLOR_MUTED);
    lv_obj_set_width(s_report_body_label, 430);
    lv_label_set_long_mode(s_report_body_label, LV_LABEL_LONG_WRAP);
}

static void edgetalk_ui_show_health_report_page(void)
{
    lv_obj_t *root = lv_screen_active();
    lv_obj_t *button;
    lv_obj_t *label;
    lv_obj_t *panel;
    lv_obj_t *body;

    reset_main_page_state();
    s_main_page_ready = 0U;
    uvc_display_set_viewport(0U, 0U, 1U, 1U);
    uvc_display_clear_viewport(0x0000U);

    lv_obj_clean(root);
    lv_obj_set_style_bg_color(root, lv_color_hex(UI_COLOR_BG), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(root, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_clear_flag(root, LV_OBJ_FLAG_SCROLLABLE);

    create_label(root, "健康报告", 24, 28, UI_FONT_CN_16, UI_COLOR_TEXT);
    create_label(root, "最终总结页面", 24, 68, UI_FONT_CN_16, UI_COLOR_CYAN);

    button = lv_button_create(root);
    lv_obj_set_pos(button, 352, 24);
    lv_obj_set_size(button, 124, 34);
    lv_obj_set_style_bg_color(button, lv_color_hex(UI_COLOR_PANEL_2), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(button, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_border_color(button, lv_color_hex(UI_COLOR_CYAN), LV_PART_MAIN);
    lv_obj_set_style_border_width(button, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(button, 8, LV_PART_MAIN);
    lv_obj_set_style_pad_all(button, 0, LV_PART_MAIN);
    lv_obj_add_event_cb(button, health_report_back_event_cb, LV_EVENT_CLICKED, NULL);

    label = create_label(button, "返回", 0, 0, UI_FONT_CN_16, UI_COLOR_CYAN);
    lv_obj_center(label);

    panel = create_panel(root, 24, 118, 464, 600);
    create_label(panel, "报告预览", 0, 0, UI_FONT_CN_16, UI_COLOR_TEXT);

    body = create_label(panel,
                        "本次健康参考报告\n"
                        "参数概览：HR 76 bpm | SpO2 98% | Circ. 正常\n"
                        "血管清晰度：显示清晰，采集质量较好\n\n"
                        "综合参考\n"
                        "当前三项参数组合表现较平稳。HR 处于常见静息范围，SpO2 表现良好，Circ. 显示本次手背静脉图像较清晰，整体结果具备较好的展示参考价值。\n"
                        "采集质量\n"
                        "本次图像中静脉纹理可见性较好，画面质量适合进行血管清晰度参考。若手部保持稳定，后续多次采集结果会更容易形成趋势对比。\n"
                        "趋势建议\n"
                        "建议在每天相近时间、相近姿势和相近光照下重复采集，用于观察个人状态变化趋势。单次结果可能受到运动、情绪、手部温度和摄像头角度影响。\n"
                        "温和建议\n"
                        "可继续保持规律作息、适量饮水和适度活动。若测量前刚运动或情绪紧张，建议静息数分钟后再查看参考结果。\n"
                        "说明\n"
                        "本报告仅用于健康状态展示和趋势参考，不作为医学诊断、治疗建议或疾病判断依据。如出现持续不适，应结合专业设备检测并咨询专业人员。",
                        0, 44, UI_FONT_CN_16, UI_COLOR_MUTED);
    lv_obj_set_width(body, 430);
    lv_label_set_long_mode(body, LV_LABEL_LONG_WRAP);
}

static void edgetalk_ui_show_main_page(void)
{
    lv_obj_t *root = lv_screen_active();

    reset_main_page_state();
    s_main_page_ready = 0U;

    lv_obj_clean(root);
    lv_obj_set_style_bg_color(root, lv_color_hex(UI_COLOR_BG), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(root, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_clear_flag(root, LV_OBJ_FLAG_SCROLLABLE);

    create_vitals_button(root);
    create_status_pill(root, "Wi-Fi --", 214, UI_COLOR_YELLOW);
    create_health_report_button(root);

    create_vein_preview(root);
    create_report(root);

    uvc_display_set_viewport((uint16_t)(VEIN_PREVIEW_PANEL_X + VEIN_PREVIEW_PAD + VEIN_PREVIEW_FRAME_X),
                             (uint16_t)(VEIN_PREVIEW_PANEL_Y + VEIN_PREVIEW_PAD + VEIN_PREVIEW_FRAME_Y),
                             VEIN_PREVIEW_FRAME_W,
                             VEIN_PREVIEW_FRAME_H);
    uvc_display_clear_viewport(0x0000U);

    s_main_page_ready = 1U;
}

static void start_button_event_cb(lv_event_t *event)
{
    LV_UNUSED(event);
    edgetalk_ui_show_main_page();
}

static void create_start_tag(lv_obj_t *parent, const char *text, int32_t x)
{
    lv_obj_t *tag = lv_obj_create(parent);
    lv_obj_t *label;

    lv_obj_set_pos(tag, x, 696);
    lv_obj_set_size(tag, 132, 32);
    lv_obj_set_style_bg_color(tag, lv_color_hex(0x122333), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(tag, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_border_color(tag, lv_color_hex(UI_COLOR_CYAN), LV_PART_MAIN);
    lv_obj_set_style_border_width(tag, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(tag, 16, LV_PART_MAIN);
    lv_obj_set_style_pad_all(tag, 0, LV_PART_MAIN);
    lv_obj_clear_flag(tag, LV_OBJ_FLAG_SCROLLABLE);

    label = create_label(tag, text, 0, 0, &lv_font_montserrat_12, UI_COLOR_CYAN);
    lv_obj_center(label);
}

static void create_start_art(lv_obj_t *root)
{
    lv_obj_t *card;
    lv_obj_t *logo;

    card = lv_obj_create(root);
    lv_obj_set_pos(card, 54, 174);
    lv_obj_set_size(card, 404, 318);
    lv_obj_set_style_bg_color(card, lv_color_hex(0x0c1a25), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(card, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_border_color(card, lv_color_hex(0x275469), LV_PART_MAIN);
    lv_obj_set_style_border_width(card, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(card, 22, LV_PART_MAIN);
    lv_obj_set_style_pad_all(card, 0, LV_PART_MAIN);
    lv_obj_clear_flag(card, LV_OBJ_FLAG_SCROLLABLE);

    create_label(card, "医疗 AI 视觉", 26, 22, UI_FONT_CN_16, UI_COLOR_MUTED);

    logo = lv_image_create(card);
    lv_image_set_src(logo, &veinsense_start_logo);
    lv_obj_set_pos(logo, 112, 58);

    create_label(card, "Ethos-U55 实时静脉预览", 92, 268,
                 UI_FONT_CN_16, UI_COLOR_MUTED);
}

static void edgetalk_ui_show_start_page(void)
{
    lv_obj_t *root = lv_screen_active();
    lv_obj_t *button;
    lv_obj_t *label;

    reset_main_page_state();
    s_main_page_ready = 0U;
    /*
     * Keep UVC display initialized without allowing its startup pattern or
     * camera frames to cover the launch page before the user enters the app.
     */
    uvc_display_set_viewport(0U, 0U, 1U, 1U);

    lv_obj_clean(root);
    lv_obj_set_style_bg_color(root, lv_color_hex(UI_COLOR_DEEP), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(root, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_clear_flag(root, LV_OBJ_FLAG_SCROLLABLE);

    create_label(root, "VeinSense AI", 46, 58, &lv_font_montserrat_24, UI_COLOR_TEXT);
    create_label(root, "Hand Vein Wellness Reference", 46, 98, &lv_font_montserrat_16, UI_COLOR_CYAN);
    create_label(root, "Tiny U-Net Vein Display | Ethos-U55",
                 46, 126, &lv_font_montserrat_16, UI_COLOR_MUTED);

    create_start_art(root);

    button = lv_button_create(root);
    lv_obj_set_pos(button, 76, 548);
    lv_obj_set_size(button, 360, 62);
    lv_obj_set_style_bg_color(button, lv_color_hex(UI_COLOR_CYAN), LV_PART_MAIN);
    lv_obj_set_style_bg_opa(button, LV_OPA_COVER, LV_PART_MAIN);
    lv_obj_set_style_border_color(button, lv_color_hex(0x9af2ff), LV_PART_MAIN);
    lv_obj_set_style_border_width(button, 1, LV_PART_MAIN);
    lv_obj_set_style_radius(button, 18, LV_PART_MAIN);
    lv_obj_add_event_cb(button, start_button_event_cb, LV_EVENT_CLICKED, NULL);

    label = create_label(button, "Start Analysis", 0, 0, &lv_font_montserrat_16, 0x071219);
    lv_obj_center(label);

    create_label(root, "点击进入实时静脉预览", 152, 626, UI_FONT_CN_16, UI_COLOR_MUTED);
    create_start_tag(root, "UVC Camera", 46);
    create_start_tag(root, "Ethos-U55", 190);
    create_start_tag(root, "Tiny U-Net", 334);
}

void edgetalk_ui_init(void)
{
    edgetalk_ui_show_start_page();
}

int edgetalk_ui_main_page_ready(void)
{
    return s_main_page_ready ? 1 : 0;
}

void edgetalk_ui_set_ai_report(const char *text)
{
    if (s_report_body_label == NULL)
    {
        return;
    }

    lv_label_set_text(s_report_body_label, (text != NULL) ? text : "");
}

static void ai_report_async_cb(void *user_data)
{
    ai_report_async_msg_t *msg = (ai_report_async_msg_t *)user_data;

    if (msg == NULL)
    {
        return;
    }

    if (msg->success)
    {
        edgetalk_ui_voice_result_ready(msg->text);
    }
    else
    {
        edgetalk_ui_voice_error(msg->text);
    }

    rt_free(msg);
}

static void post_ai_report_async(uint8_t success, const char *text)
{
    ai_report_async_msg_t *msg;

    msg = rt_calloc(1, sizeof(*msg));
    if (msg == NULL)
    {
        return;
    }

    msg->success = success ? 1U : 0U;
    rt_strncpy(msg->text, text ? text : "", sizeof(msg->text) - 1);
    msg->text[sizeof(msg->text) - 1] = '\0';

    if (lv_async_call(ai_report_async_cb, msg) != LV_RESULT_OK)
    {
        ai_report_async_cb(msg);
    }
}

void edgetalk_ui_post_ai_reply(const char *text)
{
    post_ai_report_async(1U, text ? text : "Cloud AI reply received.");
}

void edgetalk_ui_post_ai_error(const char *text)
{
    post_ai_report_async(0U, text ? text : "Cloud AI failed.");
}

void edgetalk_ui_voice_result_ready(const char *text)
{
    set_voice_state(VOICE_UI_RESULT_READY);
    if (s_report_status_label != NULL)
    {
        lv_label_set_text(s_report_status_label, "完成");
        lv_obj_set_style_text_color(s_report_status_label, lv_color_hex(UI_COLOR_GREEN), LV_PART_MAIN);
    }
    edgetalk_ui_set_ai_report((text != NULL) ? text : "Cloud AI reply received.");
}

void edgetalk_ui_voice_error(const char *text)
{
    set_voice_state(VOICE_UI_ERROR);
    if (s_report_status_label != NULL)
    {
        lv_label_set_text(s_report_status_label, "错误");
        lv_obj_set_style_text_color(s_report_status_label, lv_color_hex(UI_COLOR_RED), LV_PART_MAIN);
    }
    edgetalk_ui_set_ai_report(text);
}

void edgetalk_ui_set_circulation_score(uint8_t score)
{
    char text[8];
    uint8_t display_score;

    if (s_circulation_score_label == NULL)
    {
        return;
    }

    display_score = calibrate_circulation_score(score);

    rt_snprintf(text, sizeof(text), "%u", (unsigned)display_score);
    lv_label_set_text(s_circulation_score_label, text);
}

void edgetalk_ui_clear_circulation_score(void)
{
    if (s_circulation_score_label == NULL)
    {
        return;
    }

    lv_label_set_text(s_circulation_score_label, "--");
}

