#include <rtthread.h>
#include <rtdevice.h>
#include <webclient.h>
#include <stdlib.h>
#include <string.h>

#include "lvgl.h"
#include "edgetalk_ui.h"

#define VOICE_ASR_HEADER_SIZE       1024
#define VOICE_ASR_MAX_URL_LEN       160
#define VOICE_ASR_SAMPLE_RATE       16000
#define VOICE_ASR_SAMPLE_BITS       16
#define VOICE_ASR_CHANNELS          1
#define VOICE_ASR_READ_CHUNK        1024
#define VOICE_ASR_MAX_SECONDS       8
#define VOICE_ASR_DEFAULT_SECONDS   3
#define VOICE_ASR_MIN_BYTES         8000
#define VOICE_ASR_MAX_TEXT_LEN      512
#define VOICE_AI_REPLY_MAX_LEN      768

#ifndef VOICE_ASR_DEFAULT_URL
#define VOICE_ASR_DEFAULT_URL       "http://<PC_IP>:8000/asr"
#endif

#ifndef VOICE_ASR_MIC_DEVICE_NAME
#define VOICE_ASR_MIC_DEVICE_NAME   "mic0"
#endif

typedef enum
{
    VOICE_STATE_IDLE = 0,
    VOICE_STATE_RECORDING,
    VOICE_STATE_PROCESSING
} voice_state_t;

typedef struct
{
    int success;
    char text[VOICE_AI_REPLY_MAX_LEN];
} voice_ui_msg_t;

static char g_asr_url[VOICE_ASR_MAX_URL_LEN] = VOICE_ASR_DEFAULT_URL;
static volatile int g_voice_stop_requested;
static volatile voice_state_t g_voice_state = VOICE_STATE_IDLE;
static rt_thread_t g_voice_thread = RT_NULL;

extern int rt_hw_pdm_init(void) __attribute__((weak));
extern int ai_chat_send_text_result(const char *text, char *reply, size_t reply_size);

static void voice_keep_pdm_driver_linked(void)
{
    volatile void *pdm_init = (void *)rt_hw_pdm_init;
    (void)pdm_init;
}

static void voice_ui_async_cb(void *user_data)
{
    voice_ui_msg_t *msg = (voice_ui_msg_t *)user_data;

    if (!msg)
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

static void voice_ui_post(int success, const char *text)
{
    voice_ui_msg_t *msg;

    msg = rt_calloc(1, sizeof(*msg));
    if (!msg)
    {
        return;
    }

    msg->success = success;
    rt_strncpy(msg->text, text ? text : "", sizeof(msg->text) - 1);
    msg->text[sizeof(msg->text) - 1] = '\0';

    if (lv_async_call(voice_ui_async_cb, msg) != LV_RESULT_OK)
    {
        voice_ui_async_cb(msg);
    }
}

static void voice_print_devices(void)
{
    struct rt_object_information *info;
    struct rt_list_node *node;
    rt_base_t level;
    int count = 0;

    info = rt_object_get_information(RT_Object_Class_Device);
    if (!info)
    {
        rt_kprintf("voice_dev: no device object information\r\n");
        return;
    }

    rt_kprintf("device list:\r\n");
    level = rt_hw_interrupt_disable();
    for (node = info->object_list.next; node != &info->object_list; node = node->next)
    {
        struct rt_object *obj = rt_list_entry(node, struct rt_object, list);
        struct rt_device *dev = (struct rt_device *)obj;

        rt_hw_interrupt_enable(level);
        rt_kprintf("  %-8.*s type=%d ref=%d\r\n",
                   RT_NAME_MAX, dev->parent.name, dev->type, dev->ref_count);
        count++;
        level = rt_hw_interrupt_disable();
    }
    rt_hw_interrupt_enable(level);

    rt_kprintf("device count: %d\r\n", count);
    rt_kprintf("mic0: %s\r\n", rt_device_find(VOICE_ASR_MIC_DEVICE_NAME) ? "found" : "not found");
}

static int voice_dev(int argc, char **argv)
{
    (void)argc;
    (void)argv;
    voice_keep_pdm_driver_linked();
    voice_print_devices();
    return 0;
}
MSH_CMD_EXPORT(voice_dev, list devices and show mic0 status);

static int voice_pdm_init(int argc, char **argv)
{
    rt_device_t mic;

    (void)argc;
    (void)argv;

    mic = rt_device_find(VOICE_ASR_MIC_DEVICE_NAME);
    if (mic)
    {
        rt_kprintf("voice_pdm_init: %s already exists\r\n", VOICE_ASR_MIC_DEVICE_NAME);
        return 0;
    }

    if (!rt_hw_pdm_init)
    {
        rt_kprintf("voice_pdm_init: PDM driver is not linked into firmware\r\n");
        return -RT_ERROR;
    }

    rt_kprintf("voice_pdm_init: call rt_hw_pdm_init()\r\n");
    rt_kprintf("voice_pdm_init: ret=%d\r\n", rt_hw_pdm_init());
    voice_print_devices();
    return 0;
}
MSH_CMD_EXPORT(voice_pdm_init, manually initialize PDM microphone device);

static void voice_copy_arg(char *dst, size_t dst_size, const char *src)
{
    if (!dst || dst_size == 0)
    {
        return;
    }

    if (!src)
    {
        dst[0] = '\0';
        return;
    }

    rt_strncpy(dst, src, dst_size - 1);
    dst[dst_size - 1] = '\0';
}

int voice_asr_set_host(const char *host)
{
    if (!host || host[0] == '\0')
    {
        return -RT_ERROR;
    }

    if (!rt_strncmp(host, "http://", 7) || !rt_strncmp(host, "https://", 8))
    {
        voice_copy_arg(g_asr_url, sizeof(g_asr_url), host);
    }
    else
    {
        rt_snprintf(g_asr_url, sizeof(g_asr_url), "http://%s:8000/asr", host);
    }

    rt_kprintf("ASR URL: %s\r\n", g_asr_url);
    return RT_EOK;
}

static int voice_cfg(int argc, char **argv)
{
    if (argc == 1)
    {
        rt_kprintf("asr url: %s\r\n", g_asr_url);
        rt_kprintf("usage  : voice_cfg url <http://host:port/asr>\r\n");
        return 0;
    }

    if (argc < 3)
    {
        rt_kprintf("usage: voice_cfg <url> <value>\r\n");
        return -1;
    }

    if (!rt_strcmp(argv[1], "url"))
    {
        voice_copy_arg(g_asr_url, sizeof(g_asr_url), argv[2]);
        rt_kprintf("ok\r\n");
        return 0;
    }

    rt_kprintf("unknown config: %s\r\n", argv[1]);
    return -1;
}
MSH_CMD_EXPORT(voice_cfg, configure voice ASR server url);

static int asrurl(int argc, char **argv)
{
    if (argc < 2)
    {
        rt_kprintf("usage: asrurl <ip-or-url>\r\n");
        rt_kprintf("now  : %s\r\n", g_asr_url);
        return -1;
    }

    return voice_asr_set_host(argv[1]);
}
MSH_CMD_EXPORT(asrurl, short command: set ASR host/url);

static rt_err_t voice_config_mic(rt_device_t mic)
{
    struct rt_audio_caps caps;

    rt_memset(&caps, 0, sizeof(caps));
    caps.main_type = AUDIO_TYPE_INPUT;
    caps.sub_type = AUDIO_DSP_PARAM;
    caps.udata.config.samplerate = VOICE_ASR_SAMPLE_RATE;
    caps.udata.config.channels = VOICE_ASR_CHANNELS;
    caps.udata.config.samplebits = VOICE_ASR_SAMPLE_BITS;

    return rt_device_control(mic, AUDIO_CTL_CONFIGURE, &caps);
}

static int voice_try_init_mic(void)
{
    voice_keep_pdm_driver_linked();

    if (rt_device_find(VOICE_ASR_MIC_DEVICE_NAME))
    {
        return RT_EOK;
    }

    if (rt_hw_pdm_init)
    {
        (void)rt_hw_pdm_init();
    }

    return rt_device_find(VOICE_ASR_MIC_DEVICE_NAME) ? RT_EOK : -RT_ERROR;
}

static int voice_record_pcm_until_stop(rt_uint8_t *pcm, int pcm_size, int fixed_bytes)
{
    rt_device_t mic;
    int total = 0;

    if (voice_try_init_mic() != RT_EOK)
    {
        rt_kprintf("voice_asr: can not find %s, check BSP_USING_AUDIO_RECORD\r\n",
                   VOICE_ASR_MIC_DEVICE_NAME);
        return -RT_ERROR;
    }

    mic = rt_device_find(VOICE_ASR_MIC_DEVICE_NAME);
    if (voice_config_mic(mic) != RT_EOK)
    {
        rt_kprintf("voice_asr: configure %s failed\r\n", VOICE_ASR_MIC_DEVICE_NAME);
        return -RT_ERROR;
    }

    if (rt_device_open(mic, RT_DEVICE_OFLAG_RDONLY) != RT_EOK)
    {
        rt_kprintf("voice_asr: open %s failed\r\n", VOICE_ASR_MIC_DEVICE_NAME);
        return -RT_ERROR;
    }

    rt_kprintf("voice_asr: recording...\r\n");
    while (total < pcm_size)
    {
        int want = pcm_size - total;
        int read_len;

        if (!fixed_bytes && g_voice_stop_requested && total >= VOICE_ASR_MIN_BYTES)
        {
            break;
        }

        if (want > VOICE_ASR_READ_CHUNK)
        {
            want = VOICE_ASR_READ_CHUNK;
        }

        read_len = rt_device_read(mic, 0, pcm + total, want);
        if (read_len > 0)
        {
            total += read_len;
        }
        else
        {
            rt_thread_mdelay(10);
        }
    }

    rt_device_close(mic);
    rt_kprintf("voice_asr: recorded %d bytes\r\n", total);
    return total;
}

static void voice_copy_asr_text(char *text, size_t text_size, const char *response, int len)
{
    int copy_len;

    if (!text || text_size == 0)
    {
        return;
    }

    text[0] = '\0';
    if (!response || len <= 0)
    {
        return;
    }

    copy_len = len;
    while (copy_len > 0 && (response[copy_len - 1] == '\r' ||
                            response[copy_len - 1] == '\n' ||
                            response[copy_len - 1] == ' '))
    {
        copy_len--;
    }

    if (copy_len >= (int)text_size)
    {
        copy_len = (int)text_size - 1;
    }

    rt_memcpy(text, response, copy_len);
    text[copy_len] = '\0';
}

static int voice_post_pcm(const rt_uint8_t *pcm, int pcm_size, char *asr_text, size_t asr_text_size)
{
    struct webclient_session *session = RT_NULL;
    void *response = RT_NULL;
    size_t resp_len = 0;
    int status;
    int read_len;

    session = webclient_session_create(VOICE_ASR_HEADER_SIZE);
    if (!session)
    {
        rt_kprintf("voice_asr: no memory for webclient session\r\n");
        return -RT_ENOMEM;
    }

    webclient_header_fields_add(session, "Content-Type: application/octet-stream\r\n");
    webclient_header_fields_add(session, "X-Audio-Format: pcm_s16le\r\n");
    webclient_header_fields_add(session, "X-Audio-Sample-Rate: %d\r\n", VOICE_ASR_SAMPLE_RATE);
    webclient_header_fields_add(session, "X-Audio-Channels: %d\r\n", VOICE_ASR_CHANNELS);
    webclient_header_fields_add(session, "Content-Length: %d\r\n", pcm_size);

    rt_kprintf("POST %s, pcm=%d bytes\r\n", g_asr_url, pcm_size);
    status = webclient_post(session, g_asr_url, pcm, pcm_size);
    if (status < 0)
    {
        rt_kprintf("voice_asr: HTTP post failed: %d\r\n", status);
        webclient_close(session);
        return -RT_ERROR;
    }

    read_len = webclient_response(session, &response, &resp_len);
    if (read_len <= 0 || !response)
    {
        rt_kprintf("voice_asr: HTTP status=%d, empty response\r\n", status);
        webclient_close(session);
        return -RT_ERROR;
    }

    rt_kprintf("voice_asr: HTTP status=%d, len=%d\r\n", status, read_len);
    rt_kprintf("ASR response:\r\n%.*s\r\n", read_len, (const char *)response);
    voice_copy_asr_text(asr_text, asr_text_size, (const char *)response, read_len);

    web_free(response);
    webclient_close(session);
    return (status == 200) ? RT_EOK : -RT_ERROR;
}

static int voice_parse_seconds(int argc, char **argv, int default_seconds, int max_seconds)
{
    int seconds = default_seconds;

    if (argc >= 2)
    {
        seconds = atoi(argv[1]);
    }

    if (seconds <= 0)
    {
        seconds = default_seconds;
    }
    if (seconds > max_seconds)
    {
        seconds = max_seconds;
    }

    return seconds;
}

static int voice_run_asr_and_chat(int seconds, int send_to_ai, char *reply, size_t reply_size)
{
    int pcm_size;
    int recorded;
    rt_uint8_t *pcm;
    char *asr_text;
    int ret;

    pcm_size = VOICE_ASR_SAMPLE_RATE * (VOICE_ASR_SAMPLE_BITS / 8) * VOICE_ASR_CHANNELS * seconds;
    pcm = rt_malloc(pcm_size);
    if (!pcm)
    {
        rt_kprintf("voice: no memory for %d bytes PCM\r\n", pcm_size);
        return -RT_ENOMEM;
    }

    asr_text = rt_calloc(1, VOICE_ASR_MAX_TEXT_LEN);
    if (!asr_text)
    {
        rt_free(pcm);
        rt_kprintf("voice: no memory for ASR text\r\n");
        return -RT_ENOMEM;
    }

    recorded = voice_record_pcm_until_stop(pcm, pcm_size, 1);
    if (recorded <= 0)
    {
        rt_free(asr_text);
        rt_free(pcm);
        return -RT_ERROR;
    }

    ret = voice_post_pcm(pcm, recorded, asr_text, VOICE_ASR_MAX_TEXT_LEN);
    rt_free(pcm);
    if (ret != RT_EOK || asr_text[0] == '\0')
    {
        rt_free(asr_text);
        return -RT_ERROR;
    }

    if (send_to_ai)
    {
        rt_kprintf("voice_ask: send to AI: %s\r\n", asr_text);
        ret = ai_chat_send_text_result(asr_text, reply, reply_size);
    }

    rt_free(asr_text);
    return ret;
}

static void voice_ui_worker(void *parameter)
{
    int pcm_size;
    int recorded;
    rt_uint8_t *pcm;
    char *asr_text;
    char *reply;
    int ret;

    (void)parameter;

    pcm_size = VOICE_ASR_SAMPLE_RATE * (VOICE_ASR_SAMPLE_BITS / 8) *
               VOICE_ASR_CHANNELS * VOICE_ASR_MAX_SECONDS;
    pcm = rt_malloc(pcm_size);
    asr_text = rt_calloc(1, VOICE_ASR_MAX_TEXT_LEN);
    reply = rt_calloc(1, VOICE_AI_REPLY_MAX_LEN);

    if (!pcm || !asr_text || !reply)
    {
        voice_ui_post(0, "Voice error: no memory.");
        goto exit;
    }

    recorded = voice_record_pcm_until_stop(pcm, pcm_size, 0);
    if (recorded < VOICE_ASR_MIN_BYTES)
    {
        voice_ui_post(0, "Recording is too short. Please try again.");
        goto exit;
    }

    g_voice_state = VOICE_STATE_PROCESSING;
    ret = voice_post_pcm(pcm, recorded, asr_text, VOICE_ASR_MAX_TEXT_LEN);
    if (ret != RT_EOK || asr_text[0] == '\0')
    {
        voice_ui_post(0, "ASR failed. Check the proxy and network.");
        goto exit;
    }

    rt_kprintf("voice_ui: ASR text: %s\r\n", asr_text);
    ret = ai_chat_send_text_result(asr_text, reply, VOICE_AI_REPLY_MAX_LEN);
    if (ret != RT_EOK || reply[0] == '\0')
    {
        voice_ui_post(0, "Cloud AI failed. Check the chat proxy.");
        goto exit;
    }

    voice_ui_post(1, reply);

exit:
    if (pcm)
    {
        rt_free(pcm);
    }
    if (asr_text)
    {
        rt_free(asr_text);
    }
    if (reply)
    {
        rt_free(reply);
    }
    g_voice_stop_requested = 0;
    g_voice_state = VOICE_STATE_IDLE;
    g_voice_thread = RT_NULL;
}

int edgetalk_voice_start_request(void)
{
    if (g_voice_state != VOICE_STATE_IDLE)
    {
        return -RT_EBUSY;
    }

    g_voice_stop_requested = 0;
    g_voice_state = VOICE_STATE_RECORDING;
    g_voice_thread = rt_thread_create("voice_ui",
                                      voice_ui_worker,
                                      RT_NULL,
                                      6144,
                                      22,
                                      10);
    if (!g_voice_thread)
    {
        g_voice_state = VOICE_STATE_IDLE;
        return -RT_ENOMEM;
    }

    rt_thread_startup(g_voice_thread);
    return RT_EOK;
}

int edgetalk_voice_stop_request(void)
{
    if (g_voice_state != VOICE_STATE_RECORDING)
    {
        return -RT_ERROR;
    }

    g_voice_stop_requested = 1;
    return RT_EOK;
}

static int voice_asr(int argc, char **argv)
{
    int seconds = voice_parse_seconds(argc, argv, VOICE_ASR_DEFAULT_SECONDS, VOICE_ASR_MAX_SECONDS);
    return voice_run_asr_and_chat(seconds, 0, RT_NULL, 0);
}
MSH_CMD_EXPORT(voice_asr, record PCM and POST to ASR server: voice_asr [seconds]);

static int voice_ask(int argc, char **argv)
{
    char *reply;
    int seconds;
    int ret;

    seconds = voice_parse_seconds(argc, argv, VOICE_ASR_DEFAULT_SECONDS, VOICE_ASR_MAX_SECONDS);
    reply = rt_calloc(1, VOICE_AI_REPLY_MAX_LEN);
    if (!reply)
    {
        return -RT_ENOMEM;
    }

    ret = voice_run_asr_and_chat(seconds, 1, reply, VOICE_AI_REPLY_MAX_LEN);
    if (ret == RT_EOK)
    {
        rt_kprintf("voice_ask AI: %s\r\n", reply);
        edgetalk_ui_post_ai_reply(reply);
    }
    else
    {
        edgetalk_ui_post_ai_error("Voice AI failed. Check ASR/chat proxy and network.");
    }
    rt_free(reply);
    return ret;
}
MSH_CMD_EXPORT(voice_ask, record voice then run ASR and chat);

static int voice_mic_test(int argc, char **argv)
{
    int seconds = 1;
    int pcm_size;
    int recorded;
    rt_uint8_t *pcm;
    rt_int16_t *samples;
    int sample_count;
    int i;

    if (argc >= 2)
    {
        seconds = atoi(argv[1]);
    }
    if (seconds <= 0)
    {
        seconds = 1;
    }
    if (seconds > 3)
    {
        seconds = 3;
    }

    pcm_size = VOICE_ASR_SAMPLE_RATE * (VOICE_ASR_SAMPLE_BITS / 8) *
               VOICE_ASR_CHANNELS * seconds;
    pcm = rt_malloc(pcm_size);
    if (!pcm)
    {
        rt_kprintf("voice_mic_test: no memory\r\n");
        return -RT_ENOMEM;
    }

    recorded = voice_record_pcm_until_stop(pcm, pcm_size, 1);
    if (recorded <= 0)
    {
        rt_free(pcm);
        return -RT_ERROR;
    }

    samples = (rt_int16_t *)pcm;
    sample_count = recorded / 2;
    rt_kprintf("voice_mic_test: first samples:");
    for (i = 0; i < sample_count && i < 16; i++)
    {
        rt_kprintf(" %d", samples[i]);
    }
    rt_kprintf("\r\n");

    rt_free(pcm);
    return 0;
}
MSH_CMD_EXPORT(voice_mic_test, record mic and print sample values);
