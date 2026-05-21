"""
services/ingest_service.py — Ingest 流程編排層
職責：串接 parser → chunking → embedding → vectordb，不含任何業務邏輯本身
      是整個 pipeline 的指揮中心

流程：
  file_bytes + filename + IngestRequest
      ↓
  parser_service.parse_file()          → list[ParsedPage]
      ↓
  chunking_service.chunk_pages()       → list[Chunk]
      ↓
  model_service.get_embeddings()       → list[list[float]]
      ↓
  zip → list[EmbeddedChunk]
      ↓
  vectordb_service.upsert()            → UpsertResult
      ↓
  IngestResponse
"""
from __future__ import annotations

from app.models.schemas import (
    Chunk, EmbeddedChunk, FileType,
    IngestRequest, IngestResponse, IngestStatus,
)
from app.services import (
    parser_service,
    chunking_service,
    model_service,
    vectordb_service,
)
from pathlib import Path

from app.core.config import settings
from app.utils.helpers import (
    compute_doc_id,
    get_file_type,
    get_logger,
    make_doc_version,
)

logger = get_logger(__name__)


def _export_markdown(pages, filename: str, doc_id: str, doc_version: str) -> None:
    """將 ParsedPage list 匯出為 Markdown 檔案，存入 settings.MD_DIR"""
    import re
    from app.services.parser_service import ParsedPage

    stem    = Path(filename).stem
    # 檔名：{原始檔名}_{doc_id}_{版號}.md
    md_name = f"{stem}_{doc_id}_{doc_version}.md"
    md_path = settings.MD_DIR / md_name

    lines = [
        f"# {stem}",
        f"",
        f"> **doc_id**：`{doc_id}`  |  **version**：`{doc_version}`  |  **source**：`{filename}`",
        f"",
        "---",
        "",
    ]

    for page in pages:
        if len(pages) > 1:
            lines.append(f"## 第 {page.page_num} 頁")
            lines.append("")

        # 圖表描述
        for desc in page.figure_desc:
            lines.append("> **📊 圖表描述**")
            lines.append(">")
            for desc_line in desc.splitlines():
                lines.append(f"> {desc_line}")
            lines.append("")

        # 正文
        if page.text.strip():
            lines.append(page.text.strip())
            lines.append("")

        # 表格
        for tbl in page.tables:
            lines.append(tbl)
            lines.append("")

    md_content = "\n".join(lines)
    md_content = re.sub(r"\n{3,}", "\n\n", md_content)

    md_path.write_text(md_content, encoding="utf-8")
    logger.info("Markdown 已儲存：%s", md_path)


def run_ingest(
    file_bytes : bytes,
    filename   : str,
    request    : IngestRequest,
) -> IngestResponse:
    """
    End-to-end ingest pipeline。
    回傳 IngestResponse；任一步驟失敗都回傳 FAILED status 而不 raise。
    """
    # ── 1. 識別文件 ───────────────────────────────────
    doc_id      = compute_doc_id(file_bytes)
    doc_version = make_doc_version(request.doc_version)
    ftype_str   = get_file_type(filename)

    logger.info("開始 Ingest：file=%s  doc_id=%s  version=%s  strategy=%s",
                filename, doc_id, doc_version, request.chunk_strategy.value)

    try:
        file_type = FileType(ftype_str)
    except ValueError:
        return IngestResponse(
            doc_id      = doc_id,
            doc_version = doc_version,
            source_file = filename,
            status      = IngestStatus.FAILED,
            message     = f"不支援的格式：{ftype_str}",
        )

    try:
        # ── 2. 解析 ───────────────────────────────────
        logger.info("[1/4] 解析文件...")
        pages = parser_service.parse_file(file_bytes, filename)
        logger.info("  → %d 個邏輯頁", len(pages))

        # ── 2.5 匯出 Markdown ─────────────────────────
        _export_markdown(pages, filename, doc_id, doc_version)

        # ── 3. Chunking ───────────────────────────────
        logger.info("[2/4] Chunking（策略=%s）...", request.chunk_strategy.value)
        chunks: list[Chunk] = chunking_service.chunk_pages(
            pages       = pages,
            doc_id      = doc_id,
            doc_version = doc_version,
            source_file = filename,
            file_type   = file_type,
            strategy    = request.chunk_strategy,
            chunk_size  = request.chunk_size,
            overlap     = request.chunk_overlap,
        )
        logger.info("  → %d 個 chunk", len(chunks))

        if not chunks:
            return IngestResponse(
                doc_id      = doc_id,
                doc_version = doc_version,
                source_file = filename,
                status      = IngestStatus.DONE,
                total_chunks= 0,
                message     = "文件解析後無有效內容",
            )

        # ── 4. Embedding ──────────────────────────────
        logger.info("[3/4] Embedding（%d chunks）...", len(chunks))
        texts   = [c.content for c in chunks]
        vectors = model_service.get_embeddings(texts)

        embedded: list[EmbeddedChunk] = [
            EmbeddedChunk(chunk=c, vector=v)
            for c, v in zip(chunks, vectors)
        ]

        # ── 5. Upsert to Milvus ───────────────────────
        logger.info("[4/4] Upsert to Milvus...")
        result = vectordb_service.upsert(embedded)

        logger.info("Ingest 完成：doc_id=%s  upserted=%d",
                    doc_id, result.upserted_count)
        return IngestResponse(
            doc_id      = doc_id,
            doc_version = doc_version,
            source_file = filename,
            status      = IngestStatus.DONE,
            total_chunks= result.upserted_count,
            message     = f"成功寫入 {result.upserted_count} 個 chunk",
        )

    except Exception as e:
        logger.exception("Ingest 失敗：%s", e)
        return IngestResponse(
            doc_id      = doc_id,
            doc_version = doc_version,
            source_file = filename,
            status      = IngestStatus.FAILED,
            message     = str(e),
        )