#include <rtthread.h>
#include <webclient.h>
#include <cJSON.h>
#include <string.h>
#include <stdlib.h>

#include "edgetalk_ui.h"

#define AI_CHAT_HEADER_SIZE     1024
#define AI_CHAT_MAX_URL_LEN     192
#define AI_CHAT_MAX_KEY_LEN     192
#define AI_CHAT_MAX_MODEL_LEN   64
#define AI_CHAT_MAX_TEXT_LEN    768
#define AI_CHAT_MAX_SYSTEM_LEN  224
#define AI_CHAT_DEFAULT_LIMIT   60
#define AI_CHAT_DEFAULT_TOKENS  160
#define AI_CHAT_MAX_LIMIT       500
#define AI_CHAT_MAX_TOKENS      1024

#ifndef AI_CHAT_DEFAULT_URL
#define AI_CHAT_DEFAULT_URL     "http://<PC_IP>:8001/chat"
#endif

#ifndef AI_CHAT_DEFAULT_MODEL
#define AI_CHAT_DEFAULT_MODEL   "qwen3-max"
#endif

#ifndef AI_CHAT_DEFAULT_KEY
#define AI_CHAT_DEFAULT_KEY     ""
#endif

static char g_ai_url[AI_CHAT_MAX_URL_LEN] = AI_CHAT_DEFAULT_URL;
static char g_ai_key[AI_CHAT_MAX_KEY_LEN] = AI_CHAT_DEFAULT_KEY;
static char g_ai_model[AI_CHAT_MAX_MODEL_LEN] = AI_CHAT_DEFAULT_MODEL;
static int g_ai_reply_limit = AI_CHAT_DEFAULT_LIMIT;
static int g_ai_max_tokens = AI_CHAT_DEFAULT_TOKENS;

extern int voice_asr_set_host(const char *host);

static void ai_copy_arg(char *dst, size_t dst_size, const char *src)
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

static char *ai_join_args(int argc, char **argv, int start)
{
    int i;
    size_t used = 0;
    char *text;

    if (argc <= start)
    {
        return RT_NULL;
    }

    text = rt_calloc(1, AI_CHAT_MAX_TEXT_LEN);
    if (!text)
    {
        return RT_NULL;
    }

    for (i = start; i < argc; i++)
    {
        size_t len = rt_strlen(argv[i]);
        size_t need = len + ((i > start) ? 1 : 0);

        if (used + need >= AI_CHAT_MAX_TEXT_LEN)
        {
            break;
        }

        if (i > start)
        {
            text[used++] = ' ';
        }
        rt_memcpy(text + used, argv[i], len);
        used += len;
    }
    text[used] = '\0';

    return text;
}

int ai_chat_set_host(const char *host)
{
    if (!host || host[0] == '\0')
    {
        return -RT_ERROR;
    }

    if (!rt_strncmp(host, "http://", 7) || !rt_strncmp(host, "https://", 8))
    {
        ai_copy_arg(g_ai_url, sizeof(g_ai_url), host);
    }
    else
    {
        rt_snprintf(g_ai_url, sizeof(g_ai_url), "http://%s:8001/chat", host);
    }

    rt_kprintf("AI URL: %s\r\n", g_ai_url);
    return RT_EOK;
}

static char *ai_build_chat_body(const char *user_text)
{
    cJSON *root = RT_NULL;
    cJSON *messages = RT_NULL;
    cJSON *system_msg = RT_NULL;
    cJSON *user_msg = RT_NULL;
    char *body = RT_NULL;
    char system_prompt[AI_CHAT_MAX_SYSTEM_LEN];

    root = cJSON_CreateObject();
    messages = cJSON_CreateArray();
    system_msg = cJSON_CreateObject();
    user_msg = cJSON_CreateObject();
    if (!root || !messages || !system_msg || !user_msg)
    {
        goto exit;
    }

    cJSON_AddStringToObject(root, "model", g_ai_model);
    if (g_ai_max_tokens > 0)
    {
        cJSON_AddNumberToObject(root, "max_tokens", g_ai_max_tokens);
    }
    cJSON_AddItemToObject(root, "messages", messages);

    if (g_ai_reply_limit > 0)
    {
        rt_snprintf(system_prompt,
                    sizeof(system_prompt),
                    "你是一款演示设备配套的健康参考助手，请勿给出医学诊断结论.回答字数要求在200字以上,限制在300字以下.",
                    g_ai_reply_limit);
    }
    else
    {
        rt_snprintf(system_prompt,
                    sizeof(system_prompt),
                    "你是一款演示设备配套的健康参考助手，请勿给出医学诊断结论.回答字数要求在200字以上,限制在300字以下.");
    }

    cJSON_AddStringToObject(system_msg, "role", "system");
    cJSON_AddStringToObject(system_msg, "content", system_prompt);
    cJSON_AddItemToArray(messages, system_msg);
    system_msg = RT_NULL;

    cJSON_AddStringToObject(user_msg, "role", "user");
    cJSON_AddStringToObject(user_msg, "content", user_text);
    cJSON_AddItemToArray(messages, user_msg);
    user_msg = RT_NULL;

    body = cJSON_PrintUnformatted(root);

exit:
    if (system_msg)
    {
        cJSON_Delete(system_msg);
    }
    if (user_msg)
    {
        cJSON_Delete(user_msg);
    }
    if (root)
    {
        cJSON_Delete(root);
    }
    return body;
}

static int ai_extract_content(const char *response, char *reply, size_t reply_size)
{
    cJSON *root;
    cJSON *choices;
    cJSON *choice;
    cJSON *message;
    cJSON *content;
    cJSON *error;

    if (reply && reply_size > 0)
    {
        reply[0] = '\0';
    }

    root = cJSON_Parse(response);
    if (!root)
    {
        rt_kprintf("AI raw response:\r\n%s\r\n", response);
        if (reply && reply_size > 0)
        {
            rt_strncpy(reply, response, reply_size - 1);
            reply[reply_size - 1] = '\0';
        }
        return -RT_ERROR;
    }

    error = cJSON_GetObjectItem(root, "error");
    if (error)
    {
        char *err = cJSON_PrintUnformatted(error);
        rt_kprintf("AI error: %s\r\n", err ? err : "(unknown)");
        if (reply && reply_size > 0)
        {
            rt_snprintf(reply, reply_size, "AI error: %s", err ? err : "unknown");
        }
        if (err)
        {
            cJSON_free(err);
        }
        cJSON_Delete(root);
        return -RT_ERROR;
    }

    choices = cJSON_GetObjectItem(root, "choices");
    choice = cJSON_IsArray(choices) ? cJSON_GetArrayItem(choices, 0) : RT_NULL;
    message = choice ? cJSON_GetObjectItem(choice, "message") : RT_NULL;
    content = message ? cJSON_GetObjectItem(message, "content") : RT_NULL;

    if (cJSON_IsString(content) && content->valuestring)
    {
        rt_kprintf("AI: %s\r\n", content->valuestring);
        if (reply && reply_size > 0)
        {
            rt_strncpy(reply, content->valuestring, reply_size - 1);
            reply[reply_size - 1] = '\0';
        }
        cJSON_Delete(root);
        return RT_EOK;
    }

    rt_kprintf("AI response parse failed, raw:\r\n%s\r\n", response);
    cJSON_Delete(root);
    return -RT_ERROR;
}

static int ai_http_post_json(const char *json_body, char *reply, size_t reply_size)
{
    struct webclient_session *session = RT_NULL;
    void *response = RT_NULL;
    size_t resp_len = 0;
    int status;
    int read_len;
    int ret;

    session = webclient_session_create(AI_CHAT_HEADER_SIZE);
    if (!session)
    {
        rt_kprintf("No memory for webclient session\r\n");
        return -RT_ENOMEM;
    }

    webclient_header_fields_add(session, "Content-Type: application/json\r\n");
    if (g_ai_key[0] != '\0')
    {
        webclient_header_fields_add(session, "Authorization: Bearer %s\r\n", g_ai_key);
    }
    webclient_header_fields_add(session, "Content-Length: %d\r\n", rt_strlen(json_body));

    rt_kprintf("POST %s\r\n", g_ai_url);
    status = webclient_post(session, g_ai_url, json_body, rt_strlen(json_body));
    if (status < 0)
    {
        rt_kprintf("HTTP post failed: %d\r\n", status);
        webclient_close(session);
        return -RT_ERROR;
    }

    read_len = webclient_response(session, &response, &resp_len);
    if (read_len <= 0 || !response)
    {
        rt_kprintf("HTTP status: %d, empty response\r\n", status);
        webclient_close(session);
        return -RT_ERROR;
    }

    rt_kprintf("HTTP status: %d, len: %d\r\n", status, read_len);
    ret = ai_extract_content((const char *)response, reply, reply_size);

    web_free(response);
    webclient_close(session);
    return (status == 200 && ret == RT_EOK) ? RT_EOK : -RT_ERROR;
}

int ai_chat_send_text_result(const char *text, char *reply, size_t reply_size)
{
    char *body;
    int ret;

    if (!text || text[0] == '\0')
    {
        rt_kprintf("ai_chat_send_text: empty message\r\n");
        return -RT_ERROR;
    }

    body = ai_build_chat_body(text);
    if (!body)
    {
        rt_kprintf("No memory for request JSON\r\n");
        return -RT_ENOMEM;
    }

    ret = ai_http_post_json(body, reply, reply_size);
    cJSON_free(body);
    return ret;
}

int ai_chat_send_text(const char *text)
{
    return ai_chat_send_text_result(text, RT_NULL, 0);
}

static int ai_cfg(int argc, char **argv)
{
    if (argc == 1)
    {
        rt_kprintf("url   : %s\r\n", g_ai_url);
        rt_kprintf("model : %s\r\n", g_ai_model);
        rt_kprintf("key   : %s\r\n", g_ai_key[0] ? "(set)" : "(empty)");
        rt_kprintf("limit : %d words\r\n", g_ai_reply_limit);
        rt_kprintf("tokens: %d\r\n", g_ai_max_tokens);
        rt_kprintf("usage : ai_cfg url <http://host:port/chat>\r\n");
        rt_kprintf("        ai_cfg model <model>\r\n");
        rt_kprintf("        ai_cfg key <api_key>\r\n");
        rt_kprintf("        ai_cfg limit <0-%d words>\r\n", AI_CHAT_MAX_LIMIT);
        rt_kprintf("        ai_cfg tokens <1-%d>\r\n", AI_CHAT_MAX_TOKENS);
        return 0;
    }

    if (argc < 3)
    {
        rt_kprintf("usage: ai_cfg <url|model|key|limit|tokens> <value>\r\n");
        return -1;
    }

    if (!rt_strcmp(argv[1], "url"))
    {
        ai_copy_arg(g_ai_url, sizeof(g_ai_url), argv[2]);
    }
    else if (!rt_strcmp(argv[1], "model"))
    {
        ai_copy_arg(g_ai_model, sizeof(g_ai_model), argv[2]);
    }
    else if (!rt_strcmp(argv[1], "key"))
    {
        ai_copy_arg(g_ai_key, sizeof(g_ai_key), argv[2]);
    }
    else if (!rt_strcmp(argv[1], "limit"))
    {
        int limit = atoi(argv[2]);

        if (limit < 0)
        {
            limit = 0;
        }
        if (limit > AI_CHAT_MAX_LIMIT)
        {
            limit = AI_CHAT_MAX_LIMIT;
        }
        g_ai_reply_limit = limit;
    }
    else if (!rt_strcmp(argv[1], "tokens"))
    {
        int tokens = atoi(argv[2]);

        if (tokens <= 0)
        {
            tokens = AI_CHAT_DEFAULT_TOKENS;
        }
        if (tokens > AI_CHAT_MAX_TOKENS)
        {
            tokens = AI_CHAT_MAX_TOKENS;
        }
        g_ai_max_tokens = tokens;
    }
    else
    {
        rt_kprintf("unknown config: %s\r\n", argv[1]);
        return -1;
    }

    rt_kprintf("ok\r\n");
    return 0;
}
MSH_CMD_EXPORT(ai_cfg, configure AI chat url/model/key);

static int aiurl(int argc, char **argv)
{
    if (argc < 2)
    {
        rt_kprintf("usage: aiurl <ip-or-url>\r\n");
        rt_kprintf("now  : %s\r\n", g_ai_url);
        return -1;
    }

    return ai_chat_set_host(argv[1]);
}
MSH_CMD_EXPORT(aiurl, short command: set AI chat host/url);

static int svr(int argc, char **argv)
{
    int ret_ai;
    int ret_asr;

    if (argc < 2)
    {
        rt_kprintf("usage: svr <computer-ip>\r\n");
        rt_kprintf("ex   : svr <PC_IP>\r\n");
        return -1;
    }

    ret_ai = ai_chat_set_host(argv[1]);
    ret_asr = voice_asr_set_host(argv[1]);
    return (ret_ai == RT_EOK && ret_asr == RT_EOK) ? RT_EOK : -RT_ERROR;
}
MSH_CMD_EXPORT(svr, short command: set ASR and AI server IP);

static int ai_chat(int argc, char **argv)
{
    char *text;
    char *reply;
    int ret;

    text = ai_join_args(argc, argv, 1);
    if (!text || text[0] == '\0')
    {
        rt_kprintf("usage: ai_chat <message>\r\n");
        if (text)
        {
            rt_free(text);
        }
        return -1;
    }

    reply = rt_calloc(1, AI_CHAT_MAX_TEXT_LEN);
    if (!reply)
    {
        rt_free(text);
        rt_kprintf("No memory for AI reply\r\n");
        return -RT_ENOMEM;
    }

    ret = ai_chat_send_text_result(text, reply, AI_CHAT_MAX_TEXT_LEN);
    if (ret == RT_EOK && reply[0] != '\0')
    {
        edgetalk_ui_post_ai_reply(reply);
    }
    else
    {
        edgetalk_ui_post_ai_error("Cloud AI failed. Check the chat proxy or model.");
    }

    rt_free(reply);
    rt_free(text);
    return ret;
}
MSH_CMD_EXPORT(ai_chat, chat with OpenAI-compatible HTTP API);
