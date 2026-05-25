"""
services/chunking_service.py — 文字切割層
職責：把 ParsedPage list 依照策略切成 Chunk list
      不含解析、embedding、DB 邏輯

策略：
  HEADER — 依標題行（# ## ### 等）切割，保留標題為 metadata.title
  SIZE   — 依 token 數切割，支援重疊（overlap）
"""
from __future__ import annotations
import re

from app.models.schemas import Chunk, ChunkMeta, ChunkStrategy, FileType
from app.services.parser_service import ParsedPage
from app.utils.helpers import (
    estimate_tokens,
    make_chunk_id,
    get_logger,
)

logger = get_logger(__name__)

# ── Header 切割 ───────────────────────────────────────────

# 支援 Markdown 標題、中文章節標題
_HEADER_PATTERNS = [
    re.compile(r"^#{1,6}\s+.+"),
    re.compile(r"^第[一二三四五六七八九十百千\d]+[章節]\s*.+"),
    re.compile(r"^[一二三四五六七八九十]+[、.．]\s*.+"),
    re.compile(r"^\d+\.\s+.+"),
    re.compile(r"^\*\*.+\*\*$"),           # **標題**
    re.compile(r"^\d+\.\s+\*\*.+\*\*"),    # 3. **標題**
]

def _is_header(line: str) -> bool:
    return any(p.match(line.strip()) for p in _HEADER_PATTERNS)


def _split_by_header(text: str) -> list[tuple[str, str]]:
    """
    依標題切割，回傳 [(title, content), ...]
    段落間無明確標題時，以 "__intro__" 作為標題
    """
    lines   = text.splitlines()
    blocks  : list[tuple[str, str]] = []
    cur_title = "__intro__"
    cur_lines : list[str] = []

    for line in lines:
        if _is_header(line):
            if cur_lines or cur_title != "__intro__":
                blocks.append((cur_title, "\n".join(cur_lines).strip()))
            cur_title = line.strip()
            cur_lines = []
        else:
            cur_lines.append(line)

    if cur_lines or cur_title != "__intro__":
        blocks.append((cur_title, "\n".join(cur_lines).strip()))

    return [(t, c) for t, c in blocks if c]


# ── Size 切割 ─────────────────────────────────────────────

def _split_by_size(text: str,
                   chunk_size: int,
                   overlap: int) -> list[str]:
    """
    以「段落」為最小切割單位（保留語義完整性），
    累積到接近 chunk_size token 時切斷；
    overlap 透過保留前一塊末尾段落實現。
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks      : list[str] = []
    cur_paras   : list[str] = []
    cur_tokens  = 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)

        # 單段超過 chunk_size → 強制按句子切
        if para_tokens > chunk_size:
            if cur_paras:
                chunks.append("\n\n".join(cur_paras))
                # 保留最後一段作為 overlap
                overlap_paras = cur_paras[-1:] if cur_paras else []
                cur_paras  = overlap_paras
                cur_tokens = sum(estimate_tokens(p) for p in cur_paras)

            sentences = re.split(r"(?<=[。！？.!?])\s*", para)
            for sent in sentences:
                s_tok = estimate_tokens(sent)
                if cur_tokens + s_tok > chunk_size and cur_paras:
                    chunks.append("\n\n".join(cur_paras))
                    cur_paras  = cur_paras[-1:] if cur_paras else []
                    cur_tokens = sum(estimate_tokens(p) for p in cur_paras)
                cur_paras.append(sent)
                cur_tokens += s_tok
            continue

        if cur_tokens + para_tokens > chunk_size and cur_paras:
            chunks.append("\n\n".join(cur_paras))
            # Overlap：保留末尾若干段落，直到累積不超過 overlap tokens
            tail : list[str] = []
            tail_tok = 0
            for p in reversed(cur_paras):
                p_tok = estimate_tokens(p)
                if tail_tok + p_tok > overlap:
                    break
                tail.insert(0, p)
                tail_tok += p_tok
            cur_paras  = tail
            cur_tokens = tail_tok

        cur_paras.append(para)
        cur_tokens += para_tokens

    if cur_paras:
        chunks.append("\n\n".join(cur_paras))

    return chunks


# ── 統一入口 ──────────────────────────────────────────────

def chunk_pages(
    pages      : list[ParsedPage],
    doc_id     : str,
    doc_version: str,
    source_file: str,
    file_type  : FileType,
    strategy   : ChunkStrategy,
    chunk_size : int,
    overlap    : int,
) -> list[Chunk]:
    """
    將 ParsedPage list 依策略切成 Chunk list。
    這是 chunking_service 唯一對外的公開函式。
    """
    chunks   : list[Chunk] = []
    global_idx = 0

    for page in pages:
        full_text = page.to_full_text()
        if not full_text.strip():
            continue

        if strategy == ChunkStrategy.HEADER:
            blocks = _split_by_header(full_text)
            for title, content in blocks:
                if not content.strip():
                    continue
                _append_chunk(chunks, content, title, page.page_num,
                               global_idx, doc_id, doc_version,
                               source_file, file_type, strategy)
                global_idx += 1

        else:  # SIZE
            sub_chunks = _split_by_size(full_text, chunk_size, overlap)
            for content in sub_chunks:
                if not content.strip():
                    continue
                _append_chunk(chunks, content, "", page.page_num,
                               global_idx, doc_id, doc_version,
                               source_file, file_type, strategy)
                global_idx += 1

    logger.info("切割完成：策略=%s  共 %d 個 chunk", strategy.value, len(chunks))
    return chunks


def _append_chunk(
    chunks      : list[Chunk],
    content     : str,
    title       : str,
    page_num    : int,
    idx         : int,
    doc_id      : str,
    doc_version : str,
    source_file : str,
    file_type   : FileType,
    strategy    : ChunkStrategy,
) -> None:
    meta = ChunkMeta(
        doc_id        = doc_id,
        doc_version   = doc_version,
        source_file   = source_file,
        file_type     = file_type,
        page          = page_num,
        chunk_index   = idx,
        title         = title,
        chunk_strategy= strategy,
    )
    chunks.append(Chunk(
        chunk_id    = make_chunk_id(doc_id, idx),
        content     = content.strip(),
        token_count = estimate_tokens(content),
        meta        = meta,
    ))
