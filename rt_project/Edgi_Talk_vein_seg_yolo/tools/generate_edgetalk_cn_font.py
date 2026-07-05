from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(r"D:\RT-ThreadStudio\workspace\Edgi_Talk_M55_USB_UVC_1")
FONT_PATH = Path(r"C:\Windows\Fonts\simhei.ttf")
OUT_PATH = PROJECT / "applications" / "edgetalk_font_cn_common_16.c"
FONT_NAME = "edgetalk_font_cn_common_16"
SIZE = 16
BPP = 4
LINE_HEIGHT = 20
BASE_LINE = 4

TEXT_PARTS = [
    "\u663e\u793a\u4f53\u5f81\u9690\u85cf\u4f53\u5f81\u5065\u5eb7\u62a5\u544a\u9759\u8109\u9884\u89c8\u62a5\u544a\u9884\u89c8",
    "\u8bed\u97f3\u505c\u6b62\u7b49\u5f85\u8046\u542c\u4e2d\u5b8c\u6210\u9519\u8bef",
    "\u70b9\u51fb\u8bed\u97f3\u6309\u94ae\u5411\u4e91\u7aefAI\u63d0\u95ee",
    "\u6b63\u5728\u8046\u542c\u518d\u6b21\u70b9\u51fb\u8bed\u97f3\u53d1\u9001\u7b49\u5f85\u4e91\u7aefAI\u56de\u590d",
    "\u8bed\u97f3\u5f55\u5236\u542f\u52a8\u5931\u8d25\u505c\u6b62\u5931\u8d25\u5df2\u6536\u5230\u4e91\u7aefAI\u56de\u590d",
    "\u6700\u7ec8\u603b\u7ed3\u9875\u9762\u6b64\u9875\u9762\u9884\u7559\u7ed9\u6700\u7ec8\u5065\u5eb7\u603b\u7ed3\u548c\u6c47\u62a5",
    "\u540e\u7eed\u53ef\u6574\u5408\u9759\u8109\u9884\u89c8\u7ed3\u679c\u4e0e\u6570\u503c\u53c2\u8003\u5206\u56de\u590d\u6587\u672c",
    "\u533b\u7597\u89c6\u89c9\u5b9e\u65f6\u9759\u8109\u9884\u89c8\u624b\u80cc\u9759\u8109\u5065\u5eb7\u53c2\u8003",
    "\u9759\u8109\u663e\u793a\u5f00\u59cb\u5206\u6790\u70b9\u51fb\u8fdb\u5165\u5b9e\u65f6\u9759\u8109\u9884\u89c8",
    "\u3002\u2026\uff1a\uff0c\u3001%-|. ",
]
TEXT = "\n".join(TEXT_PARTS)


def add_gb2312_common_chars(chars):
    for high in range(0xA1, 0xF8):
        for low in range(0xA1, 0xFF):
            try:
                text = bytes([high, low]).decode("gb2312")
            except UnicodeDecodeError:
                continue
            if len(text) == 1 and text not in "\n\r\t":
                chars.add(text)


def add_current_ui_text(chars):
    ui_path = PROJECT / "applications" / "edgetalk_ui.c"
    if not ui_path.exists():
        return

    text = ui_path.read_text(encoding="utf-8", errors="ignore")
    for ch in text:
        cp = ord(ch)
        if cp >= 0x80 and ch not in "\n\r\t":
            chars.add(ch)


def pack_4bpp(values):
    out = []
    for i in range(0, len(values), 2):
        hi = values[i] & 0x0F
        lo = values[i + 1] & 0x0F if i + 1 < len(values) else 0
        out.append((hi << 4) | lo)
    return out


def c_array_u8(values, indent="    "):
    lines = []
    for i in range(0, len(values), 12):
        chunk = values[i:i + 12]
        lines.append(indent + ", ".join(f"0x{v:x}" for v in chunk) + ",")
    return "\n".join(lines)


def c_array_u16(values, indent="    "):
    lines = []
    for i in range(0, len(values), 12):
        chunk = values[i:i + 12]
        lines.append(indent + ", ".join(f"0x{v:x}" for v in chunk) + ",")
    return "\n".join(lines)


def render_glyph(font, ch):
    if ch == " ":
        adv = max(1, int(round(font.getlength(ch) * 16)))
        return [], 0, 0, 0, 0, adv

    left, top, right, bottom = font.getbbox(ch)
    width = max(0, right - left)
    height = max(0, bottom - top)
    adv = max(1, int(round(font.getlength(ch) * 16)))

    if width == 0 or height == 0:
        return [], 0, 0, 0, 0, adv

    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    draw.text((-left, -top), ch, font=font, fill=255)

    pixels = list(image.getdata())
    packed = pack_4bpp([(p + 8) // 17 for p in pixels])

    target_top = max(1, (LINE_HEIGHT - height) // 2)
    ofs_y = (LINE_HEIGHT - BASE_LINE) - height - target_top
    return packed, width, height, left, ofs_y, adv


def main():
    chars = set(chr(code) for code in range(0x20, 0x7F))
    add_gb2312_common_chars(chars)
    add_current_ui_text(chars)
    chars.update(ch for ch in TEXT if ch not in "\n\r\t")
    chars = sorted(chars, key=ord)
    font = ImageFont.truetype(str(FONT_PATH), SIZE)

    glyph_bytes = []
    glyph_desc = [None]
    codepoints = []

    for ch in chars:
        cp = ord(ch)
        bitmap_index = len(glyph_bytes)
        packed, width, height, ofs_x, ofs_y, adv = render_glyph(font, ch)
        glyph_bytes.extend(packed)
        glyph_desc.append((bitmap_index, adv, width, height, ofs_x, ofs_y))
        codepoints.append(cp)

    range_start = min(codepoints)
    unicode_list = [cp - range_start for cp in codepoints]

    lines = []
    lines.append("/*******************************************************************************")
    lines.append(" * Common GB2312 Chinese font for Edgi-Talk UI.")
    lines.append(f" * Size: {SIZE} px")
    lines.append(f" * Line height: {LINE_HEIGHT} px")
    lines.append(f" * Base line: {BASE_LINE} px")
    lines.append(f" * Bpp: {BPP}")
    lines.append(f" * Font source: {FONT_PATH}")
    lines.append(" ******************************************************************************/")
    lines.append("")
    lines.append('#include "lvgl.h"')
    lines.append("")
    lines.append("#ifndef EDGETALK_FONT_CN_COMMON_16")
    lines.append("#define EDGETALK_FONT_CN_COMMON_16 1")
    lines.append("#endif")
    lines.append("")
    lines.append("#if EDGETALK_FONT_CN_COMMON_16")
    lines.append("")
    lines.append("static LV_ATTRIBUTE_LARGE_CONST const uint8_t glyph_bitmap[] = {")
    lines.append(c_array_u8(glyph_bytes))
    lines.append("};")
    lines.append("")
    lines.append("static const lv_font_fmt_txt_glyph_dsc_t glyph_dsc[] = {")
    lines.append("    {.bitmap_index = 0, .adv_w = 0, .box_w = 0, .box_h = 0, .ofs_x = 0, .ofs_y = 0},")
    for bitmap_index, adv, width, height, ofs_x, ofs_y in glyph_desc[1:]:
        lines.append(
            f"    {{.bitmap_index = {bitmap_index}, .adv_w = {adv}, .box_w = {width}, "
            f".box_h = {height}, .ofs_x = {ofs_x}, .ofs_y = {ofs_y}}},"
        )
    lines.append("};")
    lines.append("")
    lines.append("static const uint16_t unicode_list_0[] = {")
    lines.append(c_array_u16(unicode_list))
    lines.append("};")
    lines.append("")
    lines.append("static const lv_font_fmt_txt_cmap_t cmaps[] =")
    lines.append("{")
    lines.append("    {")
    lines.append(
        f"        .range_start = {range_start}, .range_length = {max(codepoints) - range_start + 1}, .glyph_id_start = 1,"
    )
    lines.append(
        f"        .unicode_list = unicode_list_0, .glyph_id_ofs_list = NULL, .list_length = {len(unicode_list)}, .type = LV_FONT_FMT_TXT_CMAP_SPARSE_TINY"
    )
    lines.append("    }")
    lines.append("};")
    lines.append("")
    lines.append("#if LVGL_VERSION_MAJOR == 8")
    lines.append("static lv_font_fmt_txt_glyph_cache_t cache;")
    lines.append("#endif")
    lines.append("")
    lines.append("#if LVGL_VERSION_MAJOR >= 8")
    lines.append("static const lv_font_fmt_txt_dsc_t font_dsc = {")
    lines.append("#else")
    lines.append("static lv_font_fmt_txt_dsc_t font_dsc = {")
    lines.append("#endif")
    lines.append("    .glyph_bitmap = glyph_bitmap,")
    lines.append("    .glyph_dsc = glyph_dsc,")
    lines.append("    .cmaps = cmaps,")
    lines.append("    .kern_dsc = NULL,")
    lines.append("    .kern_scale = 0,")
    lines.append("    .cmap_num = 1,")
    lines.append(f"    .bpp = {BPP},")
    lines.append("    .kern_classes = 0,")
    lines.append("    .bitmap_format = 0,")
    lines.append("#if LVGL_VERSION_MAJOR == 8")
    lines.append("    .cache = &cache")
    lines.append("#endif")
    lines.append("};")
    lines.append("")
    lines.append("#if LVGL_VERSION_MAJOR >= 8")
    lines.append(f"const lv_font_t {FONT_NAME} = {{")
    lines.append("#else")
    lines.append(f"lv_font_t {FONT_NAME} = {{")
    lines.append("#endif")
    lines.append("    .get_glyph_dsc = lv_font_get_glyph_dsc_fmt_txt,")
    lines.append("    .get_glyph_bitmap = lv_font_get_bitmap_fmt_txt,")
    lines.append(f"    .line_height = {LINE_HEIGHT},")
    lines.append(f"    .base_line = {BASE_LINE},")
    lines.append("#if !(LVGL_VERSION_MAJOR == 6 && LVGL_VERSION_MINOR == 0)")
    lines.append("    .subpx = LV_FONT_SUBPX_NONE,")
    lines.append("#endif")
    lines.append("#if LV_VERSION_CHECK(7, 4, 0) || LVGL_VERSION_MAJOR >= 8")
    lines.append("    .underline_position = -2,")
    lines.append("    .underline_thickness = 0,")
    lines.append("#endif")
    lines.append("    .dsc = &font_dsc,")
    lines.append("#if LV_VERSION_CHECK(8, 2, 0) || LVGL_VERSION_MAJOR >= 9")
    lines.append("    .fallback = NULL,")
    lines.append("#endif")
    lines.append("    .user_data = NULL,")
    lines.append("};")
    lines.append("")
    lines.append("#endif")
    lines.append("")

    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")
    print(f"glyphs={len(chars)} bytes={OUT_PATH.stat().st_size}")


if __name__ == "__main__":
    main()
