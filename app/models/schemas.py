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
    HEADER = "header"     # 依標題切割
    SIZE   = "size"       # 依字數切割


class FileType(str, Enum):
    PDF   = "pdf"
    TXT   = "txt"
    DOCX  = "docx"
    XLSX  = "xlsx"
    IMAGE = "image"       # jpg / png / webp


class IngestStatus(str, Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    DONE       = "done"
    FAILED     = "failed"


# ── Chunk ─────────────────────────────────────────────────

class ChunkMeta(BaseModel):
    """每個 Chunk 隨行的 metadata，存入 Milvus 供篩選與溯源"""
    doc_id      : str             # 文件唯一碼（SHA-256 前 16 碼）
    doc_version : str             # 文件版號（上傳時間戳）
    source_file : str             # 原始檔名
    file_type   : FileType
    page        : int   = 0       # 來源頁碼（0 = 不適用）
    chunk_index : int   = 0       # 文件內第幾個 chunk（0-based）
    title       : str   = ""      # 所屬標題（header 切法才有）
    chunk_strategy: ChunkStrategy = ChunkStrategy.SIZE
    extra       : dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """單一文字區塊，含內容與 metadata"""
    chunk_id    : str             # "{doc_id}_{chunk_index:04d}"
    content     : str
    token_count : int = 0
    meta        : ChunkMeta


# ── Embedding ─────────────────────────────────────────────

class EmbeddedChunk(BaseModel):
    """Chunk + 向量，準備寫入 Milvus"""
    chunk       : Chunk
    vector      : list[float]


# ── Ingest 請求 / 回應 ────────────────────────────────────

class IngestRequest(BaseModel):
    chunk_strategy : ChunkStrategy = ChunkStrategy.SIZE
    chunk_size     : int           = 512   # size 切法：每塊最大 token 數
    chunk_overlap  : int           = 50    # size 切法：重疊 token 數
    doc_version    : str           = ""    # 留空則自動填入 ISO 時間戳


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
    """API 回傳的 Citation 結構"""
    chunk_id    : str
    source_file : str
    page        : int
    chunk_index : int
    title       : str
    content     : str        # 原文節錄（前 300 字）
    score       : float      # 向量相似度
    rerank_score: float = 0.0


class QueryRequest(BaseModel):
    question            : str
    history             : list[dict] = []     # 對話歷史（OpenAI 格式）
    top_k               : int        = 10     # 初步檢索數量
    top_n               : int        = 3      # Rerank 後保留數量
    similarity_threshold: float      = 0.0   # 相似度門檻（低於此值的 chunk 捨棄）
    use_rerank          : bool       = False  # 是否啟用 Reranking（預設關閉）
    temperature         : float      = 0.3   # LLM 生成溫度
    max_tokens          : int        = 1500  # LLM 最大輸出 token 數
    system_prompt       : str        = ""    # 自訂系統提示詞（空白用預設）
    doc_id              : str | None = None  # 限定文件 ID（選填）
    doc_version         : str | None = None  # 限定版號（選填）


class QueryResponse(BaseModel):
    question : str
    answer   : str
    citations: list[CitationOut] = []
    history  : list[dict]        = []     # 更新後的對話歷史（供下一輪傳入）