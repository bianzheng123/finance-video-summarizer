from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pilmoji import Pilmoji

from .cached_emoji_source import CachedTwitterEmojiSource

# Twitter 风格 emoji（清晰度高），带本地磁盘缓存，避免每次渲染都打 emojicdn CDN
_EMOJI_SOURCE = CachedTwitterEmojiSource()


class TableImageRenderer:
    _TABLE_FONT_PATH = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
    _TABLE_SCALE = 2
    _TABLE_WIDTH_PADDING = 3 * _TABLE_SCALE
    _TABLE_HEIGHT_PADDING = 10 * _TABLE_SCALE
    _TABLE_TOTAL_WIDTH = 1200
    _TABLE_MIN_COL_WIDTH = 50
    _TABLE_MAX_COL_WIDTH = 900
    _TABLE_HEADER_FONT_SIZE = 18 * _TABLE_SCALE
    _TABLE_BODY_FONT_SIZE = 15 * _TABLE_SCALE

    def _wrap_text(self, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
        if not text:
            return [""]

        original_lines = text.split("\n")
        lines = []
        for line in original_lines:
            current_line = ""
            current_measure = ""  # emoji-stripped version for width measurement
            for char in line:
                test = current_line + char
                test_measure = current_measure + (char if ord(char) < 0x1f000 else "  ")
                bbox = font.getbbox(test_measure)
                if bbox is None:
                    current_line = test
                    current_measure = test_measure
                    continue
                if bbox[2] - bbox[0] > max_width - self._TABLE_HEIGHT_PADDING * 2:
                    if current_line:
                        lines.append(current_line)
                    current_line = char
                    current_measure = char if ord(char) < 0x1f000 else "  "
                else:
                    current_line = test
                    current_measure = test_measure
            if current_line:
                lines.append(current_line)
        return lines if lines else [""]

    def _row_height(self, row_data: list[str], font: ImageFont.ImageFont, col_widths: list[int]) -> int:
        max_lines = 1
        for i, cell in enumerate(row_data):
            cell_clean = str(cell) if cell else ""
            lines = self._wrap_text(cell_clean, font, col_widths[i])
            max_lines = max(max_lines, len(lines))
        return max_lines * (self._TABLE_BODY_FONT_SIZE + 4) + self._TABLE_HEIGHT_PADDING * 2

    def _get_merge_groups(self, rows: list[list[str]]) -> list[tuple[int, int, str]]:
        """
        识别第一列中连续相同的值，返回合并组信息。
        返回: [(start_idx, count, text), ...]
        例如: [(0, 2, "板块A"), (2, 1, "板块B"), (3, 3, "板块C")]
        """
        if not rows:
            return []

        merge_groups = []
        current_text = rows[0][0] if rows[0] else ""
        start_idx = 0
        count = 0

        for i, row in enumerate(rows):
            cell_text = row[0] if row else ""
            if cell_text == current_text:
                count += 1
            else:
                merge_groups.append((start_idx, count, current_text))
                current_text = cell_text
                start_idx = i
                count = 1
        merge_groups.append((start_idx, count, current_text))

        return merge_groups

    def _calculate_col_widths(self, header: list[str], rows: list[str]) -> list[int]:
        """
        计算列宽：按每列平均字符串的字符长度比例分配。
        如果单元格包含换行符，取所有行段的长度之和计算平均值。
        """
        num_cols = len(header)

        col_avg_lengths = []
        for col_idx in range(num_cols):
            all_cells = [str(header[col_idx])] if col_idx < len(header) else [""]
            for row in rows:
                if col_idx < len(row):
                    all_cells.append(str(row[col_idx]))

            total_len = 0
            count = 0
            for cell in all_cells:
                if cell:
                    lines = cell.split("\n")
                    for line in lines:
                        if line == '无':
                            continue
                        total_len += len(line)
                        count += 1
            avg_len = int(total_len / count) if count > 0 else 0
            col_avg_lengths.append(avg_len)

        total_avg_len = sum(col_avg_lengths)
        if total_avg_len > 0:
            col_widths = [int(self._TABLE_TOTAL_WIDTH * (l / total_avg_len)) for l in col_avg_lengths]
        else:
            col_widths = [self._TABLE_TOTAL_WIDTH // num_cols] * num_cols

        col_widths = [max(self._TABLE_MIN_COL_WIDTH, min(w, self._TABLE_MAX_COL_WIDTH)) for w in col_widths]

        diff = self._TABLE_TOTAL_WIDTH - sum(col_widths)
        if diff != 0 and col_widths:
            col_widths[-1] += diff

        return col_widths

    def render_table(self, header: list[str], rows: list[str], output_path: Path) -> Path:
        """
        渲染表格图片
        
        Args:
            header: 表头列表，如 ["板块", "名称", "事件", "观点"]
            rows: 数据行列表，每行是一个列表，如 [["板块1", "名称1", "事件1", "观点1"], ...]
            output_path: 输出图片路径
        
        Returns:
            输出路径
        """
        cleaned_header = [str(h) for h in header]
        cleaned_rows = [[str(cell) for cell in row] for row in rows]
        
        col_widths = self._calculate_col_widths(cleaned_header, cleaned_rows)
        
        header_font = ImageFont.truetype(self._TABLE_FONT_PATH, self._TABLE_HEADER_FONT_SIZE)
        body_font = ImageFont.truetype(self._TABLE_FONT_PATH, self._TABLE_BODY_FONT_SIZE)
        line_h_hdr = self._TABLE_HEADER_FONT_SIZE + 4
        line_h_body = self._TABLE_BODY_FONT_SIZE + 4

        h_height = self._row_height(cleaned_header, header_font, col_widths) + 4
        row_heights = [self._row_height(r, body_font, col_widths) for r in cleaned_rows]
        # 获取合并组信息（第一列连续相同值）
        merge_groups = self._get_merge_groups(cleaned_rows)
        skip_col0 = set()
        merge_cells = {}  # start_idx -> (total_height, text)
        
        for start_idx, count, text in merge_groups:
            if count > 1:
                total_height = sum(row_heights[start_idx:start_idx+count])
                merge_cells[start_idx] = (total_height, text)
                for ri in range(start_idx + 1, start_idx + count):
                    skip_col0.add(ri)
                

        total_width = self._TABLE_TOTAL_WIDTH
        total_height = h_height + sum(row_heights) + 2

        img = Image.new("RGB", (total_width, total_height), (255, 255, 255))
        draw = ImageDraw.Draw(img)

        def _text_width(text: str, font: ImageFont.ImageFont) -> int:
            measure = ""
            for char in text:
                measure += char if ord(char) < 0x1f000 else "  "
            lb = font.getbbox(measure)
            return (lb[2] - lb[0]) if lb else 0

        # 收集所有需要绘制的文本调用，再统一用 Pilmoji context manager 渲染
        text_calls = []  # list of (position, text, font, fill)

        def _draw_text(position: tuple[int, int], text: str, font: ImageFont.ImageFont, fill: tuple[int, int, int]) -> None:
            text_calls.append((position, text, font, fill))

        y = 0
        x = 0
        for i, cell in enumerate(cleaned_header):
            w = col_widths[i]
            draw.rectangle([x, y, x + w, y + h_height], fill=(46, 134, 193), outline=(213, 216, 220))
            lines = self._wrap_text(cell, header_font, w)
            total_text_h = len(lines) * line_h_hdr
            ty = y + (h_height - total_text_h) // 2
            for line in lines:
                tw = _text_width(line, header_font)
                _draw_text((x + (w - tw) // 2, ty), line, font=header_font, fill=(255, 255, 255))
                ty += line_h_hdr
            x += w
        y += h_height

        for ri, row in enumerate(cleaned_rows):
            rh = row_heights[ri]
            bg = (255, 255, 255) if ri % 2 == 0 else (248, 249, 249)
            x = 0

            for ci, cell in enumerate(row):
                w = col_widths[ci] if ci < len(col_widths) else self._TABLE_MIN_COL_WIDTH

                if ci == 0 and ri in skip_col0:
                    x += w
                    continue

                if ci == 0 and ri in merge_cells:
                    total_height, text = merge_cells[ri]
                    draw.rectangle([x, y, x + w, y + total_height], fill=bg, outline=(213, 216, 220))
                    lines = self._wrap_text(text, body_font, w)
                    total_text_h = len(lines) * line_h_body
                    ty = y + (total_height - total_text_h) // 2
                    for line in lines:
                        tw = _text_width(line, body_font)
                        _draw_text((x + (w - tw) // 2, ty), line, font=body_font, fill=(44, 62, 80))
                        ty += line_h_body
                else:
                    draw.rectangle([x, y, x + w, y + rh], fill=bg, outline=(213, 216, 220))
                    lines = self._wrap_text(cell, body_font, w)
                    total_text_h = len(lines) * line_h_body
                    ty = y + (rh - total_text_h) // 2
                    for line in lines:
                        tw = _text_width(line, body_font)
                        _draw_text((x + (w - tw) // 2, ty), line, font=body_font, fill=(44, 62, 80))
                        ty += line_h_body

                x += w
            y += rh

        with Pilmoji(img, source=_EMOJI_SOURCE) as p:
            for pos, text, font, fill in text_calls:
                p.text(pos, text, font=font, fill=fill)

        img.save(str(output_path), dpi=(300, 300))
        return output_path
