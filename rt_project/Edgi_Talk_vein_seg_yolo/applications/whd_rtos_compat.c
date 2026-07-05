#include <rtthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

#include "../libraries/components/wifi-host-driver/porting/inc/rtos/cyabs_rtos.h"

#if defined(RT_USING_FINSH)
#include <finsh.h>
#endif

cy_rslt_t cy_rtos_time_get(cy_time_t *tval)
{
    if (tval == RT_NULL)
    {
        return CY_RTOS_BAD_PARAM;
    }

    *tval = (cy_time_t)rt_tick_get_millisecond();
    return CY_RSLT_SUCCESS;
}

cy_rslt_t cy_rtos_delay_milliseconds(cy_time_t num_ms)
{
    rt_thread_mdelay((rt_int32_t)num_ms);
    return CY_RSLT_SUCCESS;
}

cy_rslt_t cy_rtos_thread_create(cy_thread_t *thread,
                                cy_thread_entry_fn_t entry_function,
                                const char *name,
                                void *stack,
                                uint32_t stack_size,
                                cy_thread_priority_t priority,
                                cy_thread_arg_t arg)
{
    rt_thread_t tid;

    (void)stack;

    if ((thread == RT_NULL) || (entry_function == RT_NULL) || (stack_size < CY_RTOS_MIN_STACK_SIZE))
    {
        return CY_RTOS_BAD_PARAM;
    }

    tid = rt_thread_create(name ? name : "cyw", entry_function, arg, stack_size, priority, 2);
    if (tid == RT_NULL)
    {
        return CY_RTOS_NO_MEMORY;
    }

    if (rt_thread_startup(tid) != RT_EOK)
    {
        rt_thread_delete(tid);
        return CY_RTOS_GENERAL_ERROR;
    }

    *thread = tid;
    return CY_RSLT_SUCCESS;
}

cy_rslt_t cy_rtos_thread_exit(void)
{
    return CY_RSLT_SUCCESS;
}

cy_rslt_t cy_rtos_semaphore_init(cy_semaphore_t *semaphore, uint32_t maxcount, uint32_t initcount)
{
    (void)maxcount;

    if (semaphore == RT_NULL)
    {
        return CY_RTOS_BAD_PARAM;
    }

    *semaphore = rt_sem_create("cyw", initcount, RT_IPC_FLAG_PRIO);
    return (*semaphore == RT_NULL) ? CY_RTOS_NO_MEMORY : CY_RSLT_SUCCESS;
}

cy_rslt_t cy_rtos_semaphore_get(cy_semaphore_t *semaphore, cy_time_t timeout_ms)
{
    rt_int32_t timeout;

    if ((semaphore == RT_NULL) || (*semaphore == RT_NULL))
    {
        return CY_RTOS_BAD_PARAM;
    }

    timeout = (timeout_ms == CY_RTOS_NEVER_TIMEOUT) ? RT_WAITING_FOREVER : rt_tick_from_millisecond(timeout_ms);
    return (rt_sem_take(*semaphore, timeout) == RT_EOK) ? CY_RSLT_SUCCESS : CY_RTOS_TIMEOUT;
}

cy_rslt_t cy_rtos_semaphore_set(cy_semaphore_t *semaphore)
{
    if ((semaphore == RT_NULL) || (*semaphore == RT_NULL))
    {
        return CY_RTOS_BAD_PARAM;
    }

    return (rt_sem_release(*semaphore) == RT_EOK) ? CY_RSLT_SUCCESS : CY_RTOS_GENERAL_ERROR;
}

cy_rslt_t cy_rtos_semaphore_deinit(cy_semaphore_t *semaphore)
{
    if ((semaphore == RT_NULL) || (*semaphore == RT_NULL))
    {
        return CY_RTOS_BAD_PARAM;
    }

    if (rt_sem_delete(*semaphore) != RT_EOK)
    {
        return CY_RTOS_GENERAL_ERROR;
    }

    *semaphore = RT_NULL;
    return CY_RSLT_SUCCESS;
}

#if defined(RT_USING_FINSH)
extern int whd_builtin_nvram_set_variant(int variant);
extern int whd_builtin_nvram_get_variant(void);
extern const char *whd_builtin_nvram_get_variant_name(void);

static void whd_res_install(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    rt_kprintf("[whd_res_install] disabled: WHD resources are built into firmware.\n");
    rt_kprintf("[whd_res_install] Do not write FAL. Run wifi_init directly.\n");
}
MSH_CMD_EXPORT(whd_res_install, Disabled; WHD resources are built into firmware);

static void whd_res_check(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    rt_kprintf("[whd_res_check] built-in resources mode enabled.\n");
    rt_kprintf("[whd_res_check] firmware=55500A1.trxcse, clm=55500A1.clm_blob, nvram[%d]=%s.\n",
               whd_builtin_nvram_get_variant(),
               whd_builtin_nvram_get_variant_name());
}
MSH_CMD_EXPORT(whd_res_check, Check WHD built-in resource mode);

static void whd_nvram_variant(int argc, char **argv)
{
    int variant;

    if (argc <= 1)
    {
        rt_kprintf("Current WHD NVRAM variant: %d (%s)\n",
                   whd_builtin_nvram_get_variant(),
                   whd_builtin_nvram_get_variant_name());
        rt_kprintf("  0: CYW55513IUBG\n");
        rt_kprintf("  1: CYW955513WLIPA\n");
        rt_kprintf("  2: CYW955513SDM2WLIPA (default)\n");
        rt_kprintf("Usage: whd_nvram_variant <0|1|2>, then run wifi_init.\n");
        return;
    }

    variant = atoi(argv[1]);
    if (whd_builtin_nvram_set_variant(variant) != 0)
    {
        rt_kprintf("Invalid WHD NVRAM variant: %d\n", variant);
        rt_kprintf("Valid variants: 0, 1, 2\n");
        return;
    }

    rt_kprintf("WHD NVRAM variant set to %d (%s). Run wifi_init now.\n",
               whd_builtin_nvram_get_variant(),
               whd_builtin_nvram_get_variant_name());
}
MSH_CMD_EXPORT(whd_nvram_variant, Select WHD built-in NVRAM variant before wifi_init);
#endif
