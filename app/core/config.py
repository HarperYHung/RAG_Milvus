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
    MD_DIR       : Path = BASE_DIR / "markdown"    # 解析後的 Markdown 存放目錄

    # ── Ollama ────────────────────────────────────────
    OLLAMA_BASE_URL    : str = "http://localhost:11434"
    VISION_MODEL       : str = "gemma3:12b"       # 多模態：圖片/表格理解 + Query LLM
    EMBEDDING_MODEL    : str = "qwen3-embedding:8b"                  # 向量嵌入（中文優化）
    RERANK_MODEL       : str = "dengcao/Qwen3-Reranker-4B:Q5_K_M"  # 重排序模型

    # ── Milvus ───────────────────────────────────────
    MILVUS_HOST        : str = "localhost"
    MILVUS_PORT        : int = 19530
    MILVUS_COLLECTION  : str = "rag_documents"
    VECTOR_DIM         : int = 4096              # qwen3-embedding 維度

    # ── Chunking 預設值 ───────────────────────────────
    DEFAULT_CHUNK_SIZE    : int = 512
    DEFAULT_CHUNK_OVERLAP : int = 50

    # ── 圖表偵測門檻 ──────────────────────────────────
    RECT_COUNT_THRESHOLD   : int   = 10
    NON_WHITE_RATIO_THRESH : float = 0.08
    TEXT_GAP_THRESHOLD     : float = 0.06

    # ── AnythingLLM ──────────────────────────────────
    ANYTHINGLLM_BASE_URL : str = "http://localhost:3001"
    ANYTHINGLLM_API_KEY  : str = ""   # Settings → Tools → API Keys

    # ── API ───────────────────────────────────────────
    API_HOST : str = "0.0.0.0"
    API_PORT : int = 8000


settings = Settings()

# 確保目錄存在
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
settings.MD_DIR.mkdir(parents=True, exist_ok=True)