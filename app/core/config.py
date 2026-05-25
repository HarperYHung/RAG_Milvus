"""
config.py — 全域設定（所有模組從這裡讀取，不寫死在各層）
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ── 專案路徑 ──────────────────────────────────────
    PROJECT_NAME : str  = "RAG Pipeline"
    BASE_DIR     : Path = Path(__file__).resolve().parents[2]
    UPLOAD_DIR   : Path = BASE_DIR / "uploads"
    LOG_DIR      : Path = BASE_DIR / "logs"
    MD_DIR       : Path = BASE_DIR / "markdown"

    # ── Ollama ────────────────────────────────────────
    OLLAMA_BASE_URL : str = "http://localhost:11434"
    VISION_MODEL    : str = "gemma4:26b"
    EMBEDDING_MODEL : str = "qwen3-embedding:8b"
    RERANK_MODEL    : str = "dengcao/Qwen3-Reranker-4B:Q5_K_M"

    # ── Milvus ────────────────────────────────────────
    MILVUS_HOST       : str = "localhost"
    MILVUS_PORT       : int = 19530
    MILVUS_COLLECTION : str = "rag_documents"
    VECTOR_DIM        : int = 4096

    # ── Chunking 預設值 ───────────────────────────────
    DEFAULT_CHUNK_SIZE    : int = 512
    DEFAULT_CHUNK_OVERLAP : int = 50

    # ── PDF 解析引擎 ──────────────────────────────────
    # "auto"       → docling → paddleocr → pdfplumber（依序嘗試）
    # "docling"    → 只用 Docling（PDF/Office/HTML 原生支援）
    # "paddleocr"  → 只用 PaddleOCR（掃描版重度場景）
    # "pdfplumber" → 只用 pdfplumber（輕量 fallback）
    PDF_PARSER : str = "auto"

    # ── Vision context window ─────────────────────────
    # 描述圖片時前後各取幾頁文字作為 context（0 = 不帶 context）
    VISION_CONTEXT_PAGES     : int = 1
    VISION_CONTEXT_MAX_CHARS : int = 800

    # ── PDF 掃描版偵測門檻 ────────────────────────────
    PDF_SCAN_MIN_CHARS : int   = 50
    PDF_SCAN_MIN_RATIO : float = 0.3

    # ── 圖片最小有效大小（bytes）─────────────────────
    IMG_MIN_BYTES : int = 1024

    # ── 表格去重最小 cell 長度 ────────────────────────
    TBL_DEDUP_MIN_LEN : int = 10

    # ── AnythingLLM ───────────────────────────────────
    ANYTHINGLLM_BASE_URL : str = "http://localhost:3001"
    ANYTHINGLLM_API_KEY  : str = ""

    # ── API ───────────────────────────────────────────
    API_HOST : str = "0.0.0.0"
    API_PORT : int = 8001


settings = Settings()

settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
settings.MD_DIR.mkdir(parents=True, exist_ok=True)