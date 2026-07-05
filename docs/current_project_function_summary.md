# Edgi-Talk 当前工程重要功能与文件目录总结

主工程路径：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1
```

## 一、系统启动与主流程

功能：
负责 RT-Thread/LVGL 启动、开始页面初始化、摄像头自动启动、外设按键冻结/恢复摄像头画面。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\main.c
```

实现内容：

1. 初始化 LVGL UI。
2. 创建启动页。
3. 等待用户点击 Start Analysis 后进入主页面。
4. 等待 UVC 摄像头 ready。
5. 启动 320x240 YUYV 摄像头预览。
6. 监听外设按键 SW2/User Button。
7. 按键触发摄像头画面冻结或恢复。
8. 冻结时触发一次 AI 静脉分割处理。

## 二、USB 摄像头采集与预览

功能：
使用免驱 USB UVC 摄像头连续采集 320x240 YUYV 图像，并送入显示和 AI 处理流程。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\board\ports\usb\usbh_uvc_app.c
```

相关文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\board\ports\usb\usbh_uvc_display.c
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\board\ports\usb\usbh_uvc_display_hook.h
```

实现内容：

1. 管理 UVC 摄像头 ready 状态。
2. 启动 UVC 视频流。
3. 从 USB 摄像头队列取出视频帧。
4. 支持 YUYV 图像格式。
5. 支持冻结/恢复画面。
6. 冻结时请求 AI 处理当前帧。
7. 非冻结状态下持续刷新摄像头预览。
8. 保留低频摄像头状态日志。

## 三、摄像头画面显示到 LCD

功能：
将摄像头 YUYV 图像转换为 RGB565，并显示到 LCD 的 Vein Preview 区域，而不是覆盖整个 UI。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\board\ports\usb\usbh_uvc_display.c
```

相关接口：

```text
uvc_display_set_viewport()
uvc_display_clear_viewport()
uvc_display_frame()
uvc_display_set_overlay_callback()
```

实现内容：

1. 初始化 LCD/GFXSS 显示相关资源。
2. 获取 LCD framebuffer。
3. 将 YUYV 转换为 RGB565。
4. 根据 LVGL 页面预留区域设置 viewport。
5. 只刷新 Vein Preview 区域。
6. 避免摄像头画面覆盖整个 UI。
7. 支持 overlay callback，用于叠加静脉 mask。
8. 支持背景同步，减少 UI 和摄像头显示冲突。

## 四、静脉分割模型推理

功能：
将摄像头采集到的 YUYV 图像送入 Tiny U-Net 160x160 int8 模型，通过 Ethos-U55/NPU 进行静脉分割推理。

主要目录：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\deepcraft_ai
```

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\deepcraft_ai\src\uvc_ai.c
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\deepcraft_ai\include\uvc_ai.h
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\deepcraft_ai\include\uvc_ai_app.h
```

模型相关目录：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\deepcraft_ai\model
```

实现内容：

1. 将 320x240 YUYV 摄像头图像预处理为 160x160 模型输入。
2. 执行 int8 Tiny U-Net 静脉分割模型。
3. 获取 160x160 单通道 mask。
4. 输出推理耗时、active pixels、q range 等日志。
5. 对 mask 做阈值过滤。
6. 去除小斑点、小连通域。
7. 生成最终显示用静脉 mask。
8. 计算 Circ. 血管清晰度参考分。

## 五、静脉效果叠加显示

功能：
将模型输出的静脉 mask 以蓝灰色半透明效果叠加到摄像头画面上。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\deepcraft_ai\include\uvc_ai_app.h
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\board\ports\usb\usbh_uvc_display.c
```

实现内容：

1. AI 推理后生成 g_uvc_ai_display_mask。
2. 注册 uvc_ai_overlay_callback。
3. 显示摄像头帧时调用 overlay callback。
4. 将 160x160 mask 按摄像头显示区域缩放。
5. 对 mask 对应区域做 RGB565 颜色混合。
6. 叠加颜色为蓝灰色，透明度较低。
7. 保证静脉效果只出现在 Vein Preview 区域。
8. 冻结画面时保留最后一次效果图。

## 六、Circ. 血管清晰度参考分

功能：
根据最终显示 mask 计算血管清晰度/静脉可见性参考分，显示在主页面 Circ. 参数中。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\deepcraft_ai\include\uvc_ai_app.h
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_ui.c
```

实现内容：

1. 统计有效静脉像素。
2. 统计连通域数量。
3. 统计最大连通域大小。
4. 结合面积、连通性和碎片情况计算分数。
5. 将原始分数校准到更适合演示的显示范围。
6. UI 显示为 Circ. 百分比。
7. 该参数是血管清晰度/静脉可见性参考分，不是医学诊断指标。

## 七、LVGL 主界面显示

功能：
实现启动页、主功能页、健康报告页、按钮、状态标签、AI Report、参数显示等 UI。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_ui.c
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_ui.h
```

相关资源：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\veinsense_start_logo.h
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\veinsense_start_logo.png
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_font_cn_common_16.c
```

实现内容：

1. 启动页 Start Analysis。
2. 主页面 Vein Preview 区域。
3. 摄像头 viewport 设置。
4. HR、SpO2、Circ. 参数显示。
5. AI Report 文本显示区域。
6. 语音按钮。
7. 健康报告按钮。
8. 健康报告页面。
9. 返回主页面按钮。
10. 中文和英文混排显示。

## 八、中文字体与中英文混排

功能：
支持 UI 中文、英文、数字和符号混合显示，避免 LVGL 内置 CJK 字库缺字。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_font_cn_common_16.c
```

字体生成脚本：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\tools\generate_edgetalk_cn_font.py
```

相关配置：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\board\ports\lvgl\lv_conf.h
```

实现内容：

1. 使用自定义常用中文字库。
2. 覆盖 GB2312 常用汉字、中文标点、ASCII 字符。
3. UI 中文 label 使用该字体。
4. HR、SpO2、Circ. 等英文参数可正常混排。
5. 停用 LVGL 自带 lv_font_simsun_16_cjk。

## 九、开始页面

功能：
作为系统上电后的项目入口页面，展示项目名称、医疗图标和 Start Analysis 按钮。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_ui.c
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\veinsense_start_logo.h
```

实现内容：

1. 显示 VeinSense AI 标题。
2. 显示医疗健康相关图标。
3. 显示项目说明文本。
4. 显示 Start Analysis 按钮。
5. 点击后进入主功能页面。
6. 进入主页面后才设置摄像头 viewport。
7. 避免上电后摄像头画面覆盖启动页。

## 十、健康报告页面

功能：
显示健康参考报告，原设计可结合 HR、SpO2、Circ. 和云端 AI 回复生成总结。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_ui.c
```

实现内容：

1. 独立健康报告页面。
2. 显示 HR、SpO2、Circ. 参数说明。
3. 显示综合参考、采集质量、趋势建议、温和建议、注意事项。
4. 保留非医学诊断说明。
5. 提供返回主页面按钮。
6. 当前版本为固定演示文本。
7. 后续可接入云端 AI 返回内容，生成动态健康报告。

## 十一、AI Report 模块

功能：
主页面中的统一文本显示区域，用于显示语音状态、云端 AI 回复、错误提示等。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_ui.c
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_ui.h
```

相关接口：

```text
edgetalk_ui_set_ai_report()
edgetalk_ui_post_ai_reply()
edgetalk_ui_post_ai_error()
edgetalk_ui_voice_result_ready()
edgetalk_ui_voice_error()
```

实现内容：

1. 显示默认提示文本。
2. 显示 Listening 状态。
3. 显示 Waiting for cloud AI response 状态。
4. 显示云端 AI 回复文本。
5. 显示语音或网络错误提示。
6. 与语音按钮合并在同一模块内。
7. 尽量扩大文本显示区域以容纳 AI 回复。

## 十二、云端 AI 文本交互

功能：
开发板通过 HTTP 将文本发送到电脑端 GPT/AI 代理，并接收云端 AI 回复。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\ai_chat.c
```

相关命令：

```text
ai_chat <message>
ai_cfg
aiurl
svr <PC_IP>
```

实现内容：

1. 配置 GPT/AI 代理地址。
2. 默认访问电脑端 8001/chat 接口。
3. 将用户文本 POST 到云端 AI 代理。
4. 解析返回文本。
5. 将回复显示到 AI Report 模块。
6. 支持串口命令测试。
7. 支持通过 svr 命令同时设置 ASR 和 AI 服务器地址。

## 十三、板端语音交互

功能：
开发板采集麦克风语音，发送到电脑端 ASR 代理识别，再把识别文本发送到 AI 代理，最后将 AI 回复显示到 UI。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\voice_interaction.c
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\voice_pdm_link.c
```

相关 UI 文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_ui.c
```

实现内容：

1. 语音按钮第一次点击开始录音。
2. 第二次点击停止录音。
3. 将 PCM 语音数据上传到 ASR 代理。
4. ASR 返回识别文本。
5. 将识别文本发送到 AI 代理。
6. 接收云端 AI 英文回复。
7. 在 AI Report 区域显示回复。
8. 出错时显示错误提示。

## 十四、WiFi 联网与一键初始化

功能：
初始化 WHD WiFi，连接指定 WiFi，并设置 ASR/GPT 代理服务器地址。

主要文件：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\voice_quick_start.c
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\whd_rtos_compat.c
```

相关资源：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\whd_builtin_resources.inc
```

相关命令：

```text
wifi_init
wifi join <WIFI_SSID> <WIFI_PASSWORD>
svr <PC_IP>
vg
```

实现内容：

1. 初始化 WHD WiFi。
2. 使用内置 WHD firmware/CLM/NVRAM 资源。
3. 避免运行时写入 FAL 分区导致卡死。
4. 支持选择 NVRAM variant。
5. 连接 WiFi。
6. 设置 ASR 和 GPT 代理 IP。
7. 提供 vg 一键命令，自动执行 WiFi 初始化、连接热点和设置服务器地址。

## 十五、HTTP、JSON 和网络依赖

功能：
为 ASR/GPT 云端代理通信提供 HTTP 请求、JSON 解析和网络能力。

主要目录：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\packages
```

相关软件包：

```text
webclient-v2.2.0
cJSON-v1.7.17
FreeRTOS_Wrapper-latest
```

网络相关 RT-Thread 组件：

```text
RT-Thread WLAN
lwIP
SAL
netdev
ping/netutils
```

实现内容：

1. HTTP POST 请求。
2. JSON 请求/响应解析。
3. socket 网络通信。
4. WiFi 网络设备管理。
5. 支持串口 ping 和网络连通性测试。

## 十六、LCD / LVGL 底层适配

功能：
提供 LVGL 显示、输入、刷新和 GFX/VG Lite 相关适配。

主要目录：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\board\ports\lvgl
```

主要文件：

```text
lv_port_disp.c
lv_port_indev.c
lv_refr.c
lv_draw_vg_lite.c
lv_draw_vg_lite_img.c
lv_vg_lite_utils.c
lv_conf.h
```

实现内容：

1. LVGL 显示驱动适配。
2. LCD framebuffer 刷新。
3. 触摸输入适配。
4. LVGL 配置。
5. VG Lite/GFX 加速相关绘制。
6. 支持主界面、启动页和报告页显示。

## 十七、整体功能链路总结

完整链路如下：

1. 上电启动 RT-Thread。
2. 初始化 LVGL。
3. 显示开始页面。
4. 点击 Start Analysis 进入主页面。
5. 设置 Vein Preview 摄像头显示区域。
6. USB UVC 摄像头开始采集 320x240 YUYV。
7. YUYV 转 RGB565 后显示到 LCD。
8. 按下外设按键冻结当前画面。
9. 当前帧送入 Tiny U-Net / Ethos-U55 静脉分割模型。
10. 生成静脉 mask。
11. 对 mask 做后处理。
12. 将蓝灰色静脉效果叠加到摄像头图像。
13. 计算 Circ. 血管清晰度参考分。
14. UI 显示 HR、SpO2、Circ.。
15. WiFi 初始化后连接电脑端代理。
16. 点击语音按钮录音。
17. 语音上传到 ASR 代理。
18. ASR 文本发送到 AI 代理。
19. 云端 AI 回复显示到 AI Report。
20. 健康报告页展示综合健康参考说明。

## 十八、最核心的文件对应关系

摄像头采集：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\board\ports\usb\usbh_uvc_app.c
```

摄像头显示：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\board\ports\usb\usbh_uvc_display.c
```

静脉模型推理：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\deepcraft_ai\src\uvc_ai.c
```

静脉后处理和叠加：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\libraries\Common\deepcraft_ai\include\uvc_ai_app.h
```

LVGL UI 页面：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_ui.c
```

系统启动和按键：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\main.c
```

云端 AI 文本交互：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\ai_chat.c
```

语音录音和语音 AI：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\voice_interaction.c
```

WiFi 一键初始化：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\voice_quick_start.c
```

WHD WiFi 兼容和资源：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\whd_rtos_compat.c
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\whd_builtin_resources.inc
```

中文字体：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\edgetalk_font_cn_common_16.c
```

启动页图标：

```text
D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1\applications\veinsense_start_logo.h
```
