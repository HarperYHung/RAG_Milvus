"""
services/parser_service.py — 文件解析層
職責：把各種格式的原始檔案轉成結構化的 ParsedPage list
      每個 ParsedPage 包含頁碼、純文字、以及圖表描述（已經過 model_service 處理）

支援格式：PDF、TXT、DOCX、XLSX、圖片（jpg/png/webp/...）
不含 chunking、embedding 邏輯。
"""
from __future__ import annotations
import io
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz          # PyMuPDF
import pdfplumber
from PIL import Image

from app.core.config import settings
from app.services.model_service import describe_image
from app.utils.helpers import get_logger

logger = get_logger(__name__)

# ── 資料結構 ──────────────────────────────────────────────

@dataclass
class ParsedPage:
    """一個「邏輯頁」的解析結果（PDF 是真實頁；其他格式可能只有第 1 頁）"""
    page_num    : int
    text        : str
    tables      : list[str] = field(default_factory=list)   # Markdown 表格字串
    figure_desc : list[str] = field(default_factory=list)   # 圖表 AI 描述

    def to_full_text(self) -> str:
        """合併文字、表格、圖表描述為單一字串（供 chunking 使用）"""
        parts = [self.text.strip()]
        for tbl in self.tables:
            parts.append(f"\n\n{tbl}")
        for desc in self.figure_desc:
            parts.append(f"\n\n> **圖表描述**\n{desc}")
        return "\n".join(p for p in parts if p.strip())


# ── 圖表自動偵測（PDF 專用）──────────────────────────────

def _get_rect_count(page: pdfplumber.page.Page) -> int:
    try:
        rects = page.rects or []
        return len([r for r in rects
                    if r.get("non_stroking_color") not in (None, (1,1,1), 1)])
    except Exception:
        return 0


def _get_non_white_ratio(pixmap: fitz.Pixmap) -> float:
    try:
        img = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
        pixels = list(img.getdata())
        nw = sum(1 for r, g, b in pixels if not (r > 240 and g > 240 and b > 240))
        return nw / len(pixels)
    except Exception:
        return 0.0


def _find_figure_y_range(page: pdfplumber.page.Page) -> tuple[float, float] | None:
    """文字稀疏區偵測 → 圖表 y 範圍（頁面高度比例）"""
    page_h = float(page.height)
    if not page_h:
        return None
    words = page.extract_words() or []
    if not words:
        return None
    ys = sorted({round((float(w["top"]) + float(w["bottom"])) / 2 / page_h, 3)
                 for w in words})
    if len(ys) < 3:
        return None
    gaps = [(ys[i+1]-ys[i], ys[i], ys[i+1]) for i in range(len(ys)-1)]
    size, y0, y1 = max(gaps, key=lambda x: x[0])
    if size < settings.TEXT_GAP_THRESHOLD:
        return None
    y_top = max(0.0, y0 - 0.01)
    y_bot = min(1.0, y1 + 0.01)
    if (y_bot - y_top) < 0.08:
        return None
    return (y_top, y_bot)


def _crop_page_figure(fitz_page: fitz.Page, y_range: tuple[float, float]) -> bytes:
    rect = fitz_page.rect
    y0, y1 = y_range
    clip = fitz.Rect(rect.x0,
                     rect.y0 + rect.height * y0,
                     rect.x1,
                     rect.y0 + rect.height * y1)
    pix = fitz_page.get_pixmap(matrix=fitz.Matrix(2.5, 2.5), clip=clip,
                                colorspace=fitz.csRGB)
    return pix.tobytes("png")


# ── 表格 → Markdown ───────────────────────────────────────

def _table_to_md(rows: list) -> str:
    if not rows:
        return ""
    cleaned = [[str(c).strip() if c else "" for c in row] for row in rows]
    valid   = [r for r in cleaned if any(r)]
    if len(valid) < 2:
        return ""
    max_c = max(len(r) for r in valid)
    for r in valid:
        r += [""] * (max_c - len(r))
    lines = [
        "| " + " | ".join(valid[0]) + " |",
        "| " + " | ".join(["---"] * max_c) + " |",
    ]
    for row in valid[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════
# 各格式解析
# ══════════════════════════════════════════════════════════

def parse_pdf(file_bytes: bytes) -> list[ParsedPage]:
    logger.info("解析 PDF（%d bytes）", len(file_bytes))
    doc_fitz = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[ParsedPage] = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for idx, plumber_page in enumerate(pdf.pages):
            page_num  = idx + 1
            fitz_page = doc_fitz[idx]

            # 文字
            raw = (plumber_page.extract_text(layout=False) or "").replace("\x00", "")
            # 移除頁眉頁碼
            raw = re.sub(r"[^\n]*第\s*\d+\s*頁[^\n]*\n?", "", raw)
            raw = raw.strip()

            # 表格
            tables   = plumber_page.extract_tables() or []
            tbl_mds  = [m for m in (_table_to_md(t) for t in tables) if m]

            # 圖表偵測
            rect_cnt = _get_rect_count(plumber_page)
            lo_pix   = fitz_page.get_pixmap(matrix=fitz.Matrix(1,1),
                                            colorspace=fitz.csRGB)
            nw_ratio = _get_non_white_ratio(lo_pix)
            has_fig  = (rect_cnt >= settings.RECT_COUNT_THRESHOLD or
                        nw_ratio  >= settings.NON_WHITE_RATIO_THRESH)

            fig_descs = []
            if has_fig:
                y_range = _find_figure_y_range(plumber_page)
                if y_range:
                    logger.debug("第 %d 頁偵測到圖表，呼叫 Vision", page_num)
                    img_bytes = _crop_page_figure(fitz_page, y_range)
                    desc      = describe_image(img_bytes,
                                               context=f"第 {page_num} 頁圖表")
                    fig_descs.append(desc)

            pages.append(ParsedPage(page_num=page_num, text=raw,
                                    tables=tbl_mds, figure_desc=fig_descs))

    doc_fitz.close()
    logger.info("PDF 解析完成：%d 頁", len(pages))
    return pages


def parse_txt(file_bytes: bytes) -> list[ParsedPage]:
    text = file_bytes.decode("utf-8", errors="replace")
    return [ParsedPage(page_num=1, text=text)]


def parse_docx(file_bytes: bytes) -> list[ParsedPage]:
    from docx import Document
    from docx.oxml.ns import qn

    doc = Document(io.BytesIO(file_bytes))
    pages: list[ParsedPage] = []

    # Word 無真實頁碼，以段落分組（每 50 段為一「邏輯頁」）
    PARA_PER_PAGE = 50
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

    # 表格
    tbl_mds = []
    for tbl in doc.tables:
        rows = [[cell.text.strip() for cell in row.cells] for row in tbl.rows]
        md   = _table_to_md(rows)
        if md:
            tbl_mds.append(md)

    # 圖片：擷取並送 Vision
    fig_descs = []
    for rel in doc.part.rels.values():
        if "image" in rel.reltype:
            try:
                img_data = rel.target_part.blob
                desc     = describe_image(img_data, context="Word 文件內嵌圖片")
                fig_descs.append(desc)
            except Exception as e:
                logger.warning("Word 圖片處理失敗：%s", e)

    for i in range(0, len(paras), PARA_PER_PAGE):
        chunk_paras = paras[i:i+PARA_PER_PAGE]
        p = ParsedPage(
            page_num=i // PARA_PER_PAGE + 1,
            text="\n".join(chunk_paras),
            tables=tbl_mds if i == 0 else [],
            figure_desc=fig_descs if i == 0 else [],
        )
        pages.append(p)

    if not pages:
        pages.append(ParsedPage(page_num=1, text=""))
    return pages


def parse_xlsx(file_bytes: bytes) -> list[ParsedPage]:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
    pages = []

    for sheet_idx, ws in enumerate(wb.worksheets):
        rows = []
        for row in ws.iter_rows(values_only=True):
            rows.append([str(c).strip() if c is not None else "" for c in row])
        # 過濾全空行
        rows = [r for r in rows if any(r)]
        if not rows:
            continue

        tbl_md = _table_to_md(rows)
        # 也提供純文字版（供 embedding 使用）
        plain  = "\n".join("\t".join(r) for r in rows)
        pages.append(ParsedPage(
            page_num=sheet_idx + 1,
            text=plain,
            tables=[tbl_md] if tbl_md else [],
        ))

    if not pages:
        pages.append(ParsedPage(page_num=1, text=""))
    return pages


def parse_image(file_bytes: bytes, filename: str = "") -> list[ParsedPage]:
    """圖片直接送 Vision 模型轉文字"""
    logger.info("解析圖片：%s", filename)
    desc = describe_image(file_bytes, context=f"圖片檔案：{filename}")
    return [ParsedPage(page_num=1, text="", figure_desc=[desc])]


# ── 統一入口 ──────────────────────────────────────────────

def parse_file(file_bytes: bytes, filename: str) -> list[ParsedPage]:
    """
    根據副檔名自動選擇解析器，回傳 ParsedPage list。
    這是 parser_service 唯一對外的公開函式。
    """
    from app.utils.helpers import get_file_type
    ftype = get_file_type(filename)

    dispatch = {
        "pdf"  : parse_pdf,
        "txt"  : parse_txt,
        "docx" : parse_docx,
        "xlsx" : parse_xlsx,
    }

    if ftype == "image":
        return parse_image(file_bytes, filename)

    parser = dispatch.get(ftype)
    if parser is None:
        raise ValueError(f"不支援的檔案格式：{Path(filename).suffix}")
    return parser(file_bytes)
