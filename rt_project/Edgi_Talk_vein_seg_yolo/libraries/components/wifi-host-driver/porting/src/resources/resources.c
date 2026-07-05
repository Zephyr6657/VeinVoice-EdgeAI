/*
 * MIT License
 *
 * Copyright (c) 2024 Evlers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * Change Logs:
 * Date         Author      Notes
 * 2024-01-27   Evlers      first implementation
 */

#include <stdint.h>
#include <string.h>

#include "rtthread.h"

#include "wiced_resource.h"
#include "whd_resource_api.h"

#define DBG_TAG           "whd.resources"
#define DBG_LVL           DBG_INFO
#include "rtdbg.h"


static const char *name_list[] =
{
#ifdef WHD_RESOURCES_IN_EXTERNAL_STORAGE_FAL
    [WHD_RESOURCE_WLAN_FIRMWARE] = WHD_RESOURCES_FIRMWARE_PART_NAME,
    [WHD_RESOURCE_WLAN_CLM] = WHD_RESOURCES_CLM_PART_NAME,
    [WHD_RESOURCE_WLAN_NVRAM] = WHD_RESOURCES_NVRAM_PART_NAME
#endif
#ifdef WHD_RESOURCES_IN_EXTERNAL_STORAGE_FS
    [WHD_RESOURCE_WLAN_FIRMWARE] = WHD_RESOURCES_FIRMWARE_PATH_NAME,
    [WHD_RESOURCE_WLAN_CLM] = WHD_RESOURCES_CLM_PATH_NAME,
    [WHD_RESOURCE_WLAN_NVRAM] = WHD_RESOURCES_NVRAM_PATH_NAME
#endif
};

uint8_t r_buffer[WHD_RESOURCES_BLOCK_SIZE];

static void nvram_convert_line_endings(char *data, uint32_t size)
{
    /* convert the newline to null-terminator */
    for (uint32_t i = 0; i < size; i ++)
    {
        if (data[i] == 0x0A)
        {
            data[i] = 0x00;
        }
    }
}

#ifdef WHD_RESOURCES_IN_EXTERNAL_STORAGE_FAL
#include "whd_builtin_resources.inc"

static int g_whd_nvram_variant = 2;
static int g_whd_nvram_log_printed;

static const struct
{
    const char *name;
    const uint8_t *data;
    uint32_t size;
} nvram_variants[] =
{
    { "CYW55513IUBG", whd_builtin_nvram_cyw55513iubg, whd_builtin_nvram_cyw55513iubg_size },
    { "CYW955513WLIPA", whd_builtin_nvram_cyw955513wlipa, whd_builtin_nvram_cyw955513wlipa_size },
    { "CYW955513SDM2WLIPA", whd_builtin_nvram_cyw955513sdm2wlipa, whd_builtin_nvram_cyw955513sdm2wlipa_size },
};

int whd_builtin_nvram_set_variant(int variant)
{
    if ((variant < 0) || (variant >= (int)(sizeof(nvram_variants) / sizeof(nvram_variants[0]))))
    {
        return -1;
    }

    g_whd_nvram_variant = variant;
    g_whd_nvram_log_printed = 0;
    return 0;
}

int whd_builtin_nvram_get_variant(void)
{
    return g_whd_nvram_variant;
}

const char *whd_builtin_nvram_get_variant_name(void)
{
    return nvram_variants[g_whd_nvram_variant].name;
}

static uint32_t get_builtin_resource(whd_resource_type_t type, const uint8_t **data, uint32_t *size)
{
    switch (type)
    {
    case WHD_RESOURCE_WLAN_FIRMWARE:
        *data = whd_builtin_firmware;
        *size = whd_builtin_firmware_size;
        return RESOURCE_SUCCESS;

    case WHD_RESOURCE_WLAN_CLM:
        *data = whd_builtin_clm;
        *size = whd_builtin_clm_size;
        return RESOURCE_SUCCESS;

    case WHD_RESOURCE_WLAN_NVRAM:
        *data = nvram_variants[g_whd_nvram_variant].data;
        *size = nvram_variants[g_whd_nvram_variant].size;
        if (!g_whd_nvram_log_printed)
        {
            LOG_I("Using built-in NVRAM variant %d: %s, size=%lu",
                  g_whd_nvram_variant,
                  nvram_variants[g_whd_nvram_variant].name,
                  (unsigned long)*size);
            g_whd_nvram_log_printed = 1;
        }
        return RESOURCE_SUCCESS;

    default:
        LOG_E("Unsupported WHD resource type %d", type);
        return RESOURCE_FILE_OPEN_FAIL;
    }
}

static uint32_t host_platform_resource_size (whd_driver_t whd_drv, whd_resource_type_t type, uint32_t *size_out)
{
    const uint8_t *resource_data;

    (void)whd_drv;
    return get_builtin_resource(type, &resource_data, size_out);
}

static uint32_t host_get_resource_block_size (whd_driver_t whd_drv, whd_resource_type_t type, uint32_t *size_out)
{
    *size_out = WHD_RESOURCES_BLOCK_SIZE;
    return RESOURCE_SUCCESS;
}

static uint32_t host_get_resource_no_of_blocks (whd_driver_t whd_drv, whd_resource_type_t type, uint32_t *block_count)
{
    uint32_t resource_size = 0;
    uint32_t block_size;

    host_platform_resource_size(whd_drv, type, &resource_size);
    host_get_resource_block_size(whd_drv, type, &block_size);
    *block_count = resource_size / block_size;
    if (resource_size % block_size)
        *block_count += 1;

    return RESOURCE_SUCCESS;
}

static uint32_t host_get_resource_block (whd_driver_t whd_drv, whd_resource_type_t type, uint32_t blockno, const uint8_t **data, uint32_t *size_out)
{
    uint32_t resource_size;
    uint32_t block_size;
    uint32_t block_count;
    uint32_t read_pos;
    uint32_t read_size;
    const uint8_t *resource_data;

    if (get_builtin_resource(type, &resource_data, &resource_size) != RESOURCE_SUCCESS)
    {
        return RESOURCE_FILE_OPEN_FAIL;
    }

    host_get_resource_block_size(whd_drv, type, &block_size);
    host_get_resource_no_of_blocks(whd_drv, type, &block_count);
    memset(r_buffer, 0, block_size);
    read_pos = blockno * block_size;

    if (blockno >= block_count)
    {
        return WHD_BADARG;
    }

    if (read_pos > resource_size)
    {
        LOG_E("Read position beyond built-in resource size");
        return RESOURCE_OFFSET_TOO_BIG;
    }

    read_size = MIN(block_size, resource_size - read_pos);
    memcpy(r_buffer, resource_data + read_pos, read_size);
    *size_out = read_size;
    *data = (uint8_t *)&r_buffer;

    if (type == WHD_RESOURCE_WLAN_NVRAM)
    {
        nvram_convert_line_endings((char *)*data, *size_out);
    }

    return RESOURCE_SUCCESS;
}

static uint32_t host_resource_read (whd_driver_t whd_drv, whd_resource_type_t type, uint32_t offset, uint32_t size, uint32_t *size_out, void *buffer)
{
    uint32_t resource_size;
    uint32_t read_size;
    const uint8_t *resource_data;

    (void)whd_drv;

    if (get_builtin_resource(type, &resource_data, &resource_size) != RESOURCE_SUCCESS)
    {
        return RESOURCE_FILE_OPEN_FAIL;
    }

    if (offset > resource_size)
    {
        LOG_E("Read offset beyond built-in resource size");
        return RESOURCE_OFFSET_TOO_BIG;
    }

    read_size = MIN(size, resource_size - offset);
    memcpy(buffer, resource_data + offset, read_size);
    *size_out = read_size;

    if (type == WHD_RESOURCE_WLAN_NVRAM)
    {
        nvram_convert_line_endings((char *)buffer, *size_out);
    }

    return RESOURCE_SUCCESS;
}
#endif

#ifdef WHD_RESOURCES_IN_EXTERNAL_STORAGE_FS
#include <dfs_file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

static uint32_t host_platform_resource_size (whd_driver_t whd_drv, whd_resource_type_t type, uint32_t *size_out)
{
    int fd;
    struct stat stat;
    const char *path_name = name_list[type];

    if ((fd = open(path_name, O_RDONLY)) < 0)
    {
        LOG_E("No %s file found", path_name);
        return RESOURCE_FILE_OPEN_FAIL;
    }

    if (fstat(fd, &stat) < 0)
    {
        close(fd);
        LOG_E("Read failed for file[%s]", path_name);
        return RESOURCE_FILE_OPEN_FAIL;
    }

    *size_out = stat.st_size;

    close(fd);

    return WHD_SUCCESS;
}

static uint32_t host_get_resource_block_size (whd_driver_t whd_drv, whd_resource_type_t type, uint32_t *size_out)
{
    *size_out = WHD_RESOURCES_BLOCK_SIZE;
    return RESOURCE_SUCCESS;
}

static uint32_t host_get_resource_no_of_blocks (whd_driver_t whd_drv, whd_resource_type_t type, uint32_t *block_count)
{
    uint32_t resource_size = 0;
    uint32_t block_size;

    host_platform_resource_size(whd_drv, type, &resource_size);
    host_get_resource_block_size(whd_drv, type, &block_size);
    *block_count = resource_size / block_size;
    if (resource_size % block_size)
        *block_count += 1;

    return RESOURCE_SUCCESS;
}

static uint32_t host_get_resource_block (whd_driver_t whd_drv, whd_resource_type_t type, uint32_t blockno, const uint8_t **data, uint32_t *size_out)
{
    int fd;
    uint32_t resource_size = 0;
    uint32_t block_size;
    uint32_t block_count;
    uint32_t read_pos;
    const char *path_name = name_list[type];

    host_platform_resource_size(whd_drv, type, &resource_size);
    host_get_resource_block_size(whd_drv, type, &block_size);
    host_get_resource_no_of_blocks(whd_drv, type, &block_count);
    memset(r_buffer, 0, block_size);
    read_pos = blockno * block_size;

    if (blockno >= block_count)
    {
        return WHD_BADARG;
    }

    if ((fd = open(path_name, O_RDONLY)) < 0)
    {
        LOG_E("No %s file found", path_name);
        return RESOURCE_FILE_OPEN_FAIL;
    }

    if (read_pos > resource_size)
    {
        close(fd);
        return RESOURCE_OFFSET_TOO_BIG;
    }

    lseek(fd, read_pos, SEEK_SET);
    *size_out = read(fd, r_buffer, MIN(block_size, resource_size - read_pos));
    *data = (uint8_t *)&r_buffer;

    if (type == WHD_RESOURCE_WLAN_NVRAM)
    {
        nvram_convert_line_endings((char *)*data, *size_out);
    }

    close(fd);

    return RESOURCE_SUCCESS;
}

static uint32_t host_resource_read (whd_driver_t whd_drv, whd_resource_type_t type, uint32_t offset, uint32_t size, uint32_t *size_out, void *buffer)
{
    int fd;
    uint32_t resource_size = 0;
    const char *path_name = name_list[type];

    host_platform_resource_size(whd_drv, type, &resource_size);

    if ((fd = open(path_name, O_RDONLY)) < 0)
    {
        LOG_E("No %s file found", path_name);
        return RESOURCE_FILE_OPEN_FAIL;
    }

    if (offset > resource_size)
    {
        close(fd);
        return RESOURCE_OFFSET_TOO_BIG;
    }

    lseek(fd, offset, SEEK_SET);
    /* read directly into the provided buffer */
    *size_out = read(fd, buffer, MIN(size, resource_size - offset));

    if (type == WHD_RESOURCE_WLAN_NVRAM)
    {
        nvram_convert_line_endings((char *)buffer, *size_out);
    }

    close(fd);

    return RESOURCE_SUCCESS;
}
#endif

whd_resource_source_t resource_ops =
{
    .whd_resource_size = host_platform_resource_size,
    .whd_get_resource_block_size = host_get_resource_block_size,
    .whd_get_resource_no_of_blocks = host_get_resource_no_of_blocks,
    .whd_get_resource_block = host_get_resource_block,
    .whd_resource_read = host_resource_read
};
