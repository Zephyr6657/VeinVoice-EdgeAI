#include <rtthread.h>
#include <stdint.h>

#if defined(BSP_USING_DEEPCRAFT_AI)

void *mtb_ml_mutex_create(void)
{
    return rt_mutex_create("mlmtx", RT_IPC_FLAG_PRIO);
}

int mtb_ml_mutex_lock(void *mutex)
{
    if (mutex == RT_NULL)
    {
        return -RT_ERROR;
    }

    return rt_mutex_take((rt_mutex_t)mutex, RT_WAITING_FOREVER);
}

int mtb_ml_mutex_unlock(void *mutex)
{
    if (mutex == RT_NULL)
    {
        return -RT_ERROR;
    }

    return rt_mutex_release((rt_mutex_t)mutex);
}

void mtb_ml_mutex_destroy(void *mutex)
{
    if (mutex != RT_NULL)
    {
        rt_mutex_delete((rt_mutex_t)mutex);
    }
}

void *mtb_ml_sem_create(void)
{
    return rt_sem_create("mlsem", 0, RT_IPC_FLAG_PRIO);
}

int mtb_ml_sem_take(void *sem, uint64_t timeout)
{
    rt_int32_t rt_timeout = RT_WAITING_FOREVER;

    if (sem == RT_NULL)
    {
        return -RT_ERROR;
    }

    if (timeout != UINT64_MAX)
    {
        rt_timeout = rt_tick_from_millisecond((rt_int32_t)timeout);
    }

    return rt_sem_take((rt_sem_t)sem, rt_timeout);
}

void mtb_ml_sem_destroy(void *sem)
{
    if (sem != RT_NULL)
    {
        rt_sem_delete((rt_sem_t)sem);
    }
}

#endif
