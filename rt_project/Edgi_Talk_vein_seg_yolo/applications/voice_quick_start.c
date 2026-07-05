#include <rtthread.h>
#include <wlan_mgnt.h>
#include <string.h>

#define VG_WIFI_SSID        "YOUR_WIFI_SSID"
#define VG_WIFI_PASSWORD    "YOUR_WIFI_PASSWORD"
#define VG_SERVER_IP        "YOUR_PC_IP"
#define VG_INIT_TIMEOUT_MS  30000
#define VG_JOIN_TIMEOUT_MS  20000
#define VG_THREAD_STACK     4096
#define VG_THREAD_PRIORITY  24

extern int whd_wifi_init_manual(void);
extern int ai_chat_set_host(const char *host);
extern int voice_asr_set_host(const char *host);

static volatile rt_bool_t g_vg_busy = RT_FALSE;

static rt_bool_t vg_sta_ready(void)
{
    return rt_device_find(RT_WLAN_DEVICE_STA_NAME) != RT_NULL;
}

static rt_bool_t vg_wait_sta_ready(rt_int32_t timeout_ms)
{
    rt_int32_t elapsed = 0;
    const rt_int32_t step = 100;

    while (elapsed < timeout_ms)
    {
        if (vg_sta_ready())
        {
            return RT_TRUE;
        }

        rt_thread_mdelay(step);
        elapsed += step;
    }

    return RT_FALSE;
}

static rt_bool_t vg_connected_to_target(void)
{
    struct rt_wlan_info info;

    if (rt_wlan_is_connected() != RT_TRUE)
    {
        return RT_FALSE;
    }

    rt_memset(&info, 0, sizeof(info));
    if (rt_wlan_get_info(&info) != RT_EOK)
    {
        return RT_FALSE;
    }

    return (info.ssid.len == rt_strlen(VG_WIFI_SSID)) &&
           (rt_memcmp(info.ssid.val, VG_WIFI_SSID, info.ssid.len) == 0);
}

static rt_bool_t vg_wait_connected(rt_int32_t timeout_ms)
{
    rt_int32_t elapsed = 0;
    const rt_int32_t step = 200;

    while (elapsed < timeout_ms)
    {
        if (vg_connected_to_target())
        {
            return RT_TRUE;
        }

        rt_thread_mdelay(step);
        elapsed += step;
    }

    return RT_FALSE;
}

static void vg_worker(void *parameter)
{
    rt_err_t ret;

    (void)parameter;

    rt_kprintf("[vg] wifi init...\n");
    if (!vg_sta_ready())
    {
        ret = whd_wifi_init_manual();
        if (ret != RT_EOK)
        {
            rt_kprintf("[vg] wifi init start failed: %d\n", ret);
            goto exit;
        }

        if (!vg_wait_sta_ready(VG_INIT_TIMEOUT_MS))
        {
            rt_kprintf("[vg] wifi init timeout\n");
            goto exit;
        }
    }
    else
    {
        rt_kprintf("[vg] wifi already initialized\n");
    }

    if (vg_connected_to_target())
    {
        rt_kprintf("[vg] already connected to %s\n", VG_WIFI_SSID);
    }
    else
    {
        rt_kprintf("[vg] join %s...\n", VG_WIFI_SSID);
        ret = rt_wlan_connect(VG_WIFI_SSID, VG_WIFI_PASSWORD);
        if (ret != RT_EOK)
        {
            rt_kprintf("[vg] join failed: %d\n", ret);
            goto exit;
        }

        if (!vg_wait_connected(VG_JOIN_TIMEOUT_MS))
        {
            rt_kprintf("[vg] join timeout\n");
            goto exit;
        }

        rt_thread_mdelay(1500);
    }

    rt_kprintf("[vg] server %s...\n", VG_SERVER_IP);
    ret = ai_chat_set_host(VG_SERVER_IP);
    if (ret != RT_EOK)
    {
        rt_kprintf("[vg] AI server set failed: %d\n", ret);
        goto exit;
    }

    ret = voice_asr_set_host(VG_SERVER_IP);
    if (ret != RT_EOK)
    {
        rt_kprintf("[vg] ASR server set failed: %d\n", ret);
        goto exit;
    }

    rt_kprintf("[vg] ready\n");

exit:
    g_vg_busy = RT_FALSE;
}

static int vg(int argc, char **argv)
{
    rt_thread_t tid;

    (void)argc;
    (void)argv;

    rt_enter_critical();
    if (g_vg_busy)
    {
        rt_exit_critical();
        rt_kprintf("[vg] busy\n");
        return 0;
    }
    g_vg_busy = RT_TRUE;
    rt_exit_critical();

    tid = rt_thread_create("vg", vg_worker, RT_NULL,
                           VG_THREAD_STACK, VG_THREAD_PRIORITY, 10);
    if (tid == RT_NULL)
    {
        g_vg_busy = RT_FALSE;
        rt_kprintf("[vg] create thread failed\n");
        return -RT_ERROR;
    }

    rt_thread_startup(tid);
    return 0;
}
MSH_CMD_EXPORT(vg, quick WiFi and voice server setup);
