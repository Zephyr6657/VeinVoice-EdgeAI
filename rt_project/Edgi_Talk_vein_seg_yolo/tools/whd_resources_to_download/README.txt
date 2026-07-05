WHD resources for Edgi_Talk_M55_USB_UVC_1

Send these files from a serial terminal with YMODEM after running the matching MSH command:

1. whd_res_download whd_firmware
   send: 55500A1.trxcse

2. whd_res_download whd_clm
   send: 55500A1.clm_blob

3. whd_res_download whd_nvram
   send: wifi_nvram_55500_CYW55513IUBG.bin

After all three downloads succeed, run wifi_init again.
