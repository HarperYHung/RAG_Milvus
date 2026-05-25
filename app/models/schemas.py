"""
models/schemas.py — 所有 Pydantic 資料結構
職責：定義跨層傳遞的資料合約，不含任何業務邏輯
"""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── 列舉 ──────────────────────────────────────────────────

class ChunkStrategy(str, Enum):
    HEADER = "header"
    SIZE   = "size"


class FileType(str, Enum):
    PDF   = "pdf"
    TXT   = "txt"
    DOCX  = "docx"
    XLSX  = "xlsx"
    PPTX  = "pptx"
    IMAGE = "image"


class ContentType(str, Enum):
    """ParsedPage 內容類型（借鑑 RAG-Anything content_list type 欄位）"""
    TEXT     = "text"      # 純文字段落
    TABLE    = "table"     # 表格
    IMAGE    = "image"     # 圖片描述
    EQUATION = "equation"  # 數學公式（MinerU 萃取）
    MIXED    = "mixed"     # 混合（預設）


class IngestStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


# ── Chunk ─────────────────────────────────────────────────

class ChunkMeta(BaseModel):
    doc_id         : str
    doc_version    : str
    source_file    : str
    file_type      : FileType
    page           : int          = 0
    chunk_index    : int          = 0
    title          : str          = ""
    chunk_strategy : ChunkStrategy = ChunkStrategy.SIZE
    # 新增：內容類型，供查詢層做模態過濾
    content_type   : ContentType  = ContentType.MIXED
    extra          : dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id    : str
    content     : str
    token_count : int = 0
    meta        : ChunkMeta


# ── Embedding ─────────────────────────────────────────────

class EmbeddedChunk(BaseModel):
    chunk  : Chunk
    vector : list[float]


# ── Ingest 請求 / 回應 ────────────────────────────────────

class IngestRequest(BaseModel):
    chunk_strategy : ChunkStrategy = ChunkStrategy.SIZE
    chunk_size     : int           = 512
    chunk_overlap  : int           = 50
    doc_version    : str           = ""


class IngestResponse(BaseModel):
    doc_id       : str
    doc_version  : str
    source_file  : str
    status       : IngestStatus
    total_chunks : int = 0
    message      : str = ""


# ── Upsert 結果 ───────────────────────────────────────────

class UpsertResult(BaseModel):
    doc_id         : str
    doc_version    : str
    upserted_count : int
    skipped_count  : int = 0
    message        : str = ""


# ── Query 請求 / 回應 ─────────────────────────────────────

class CitationOut(BaseModel):
    chunk_id     : str
    source_file  : str
    page         : int
    chunk_index  : int
    title        : str
    content      : str
    score        : float
    rerank_score : float       = 0.0
    content_type : ContentType = ContentType.MIXED


class QueryRequest(BaseModel):
    question             : str
    history              : list[dict] = []
    top_k                : int        = 10
    top_n                : int        = 3
    similarity_threshold : float      = 0.0
    use_rerank           : bool       = False
    temperature          : float      = 0.3
    max_tokens           : int        = 1500
    system_prompt        : str        = ""
    doc_id               : str | None = None
    doc_version          : str | None = None
    # 新增：可限定只檢索特定模態（None = 全部）
    content_type         : ContentType | None = None


class QueryResponse(BaseModel):
    question  : str
    answer    : str
    citations : list[CitationOut] = []
    history   : list[dict]        = []