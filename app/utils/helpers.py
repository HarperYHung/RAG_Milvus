"""
utils/helpers.py — 純工具函式，不依賴任何業務層
"""
import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import settings


# ── Logger ────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Console
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # File
        fh = logging.FileHandler(settings.LOG_DIR / "rag.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ── 文件唯一碼 ────────────────────────────────────────────

def compute_doc_id(file_bytes: bytes) -> str:
    """SHA-256 取前 16 碼作為文件唯一碼，內容不變則 ID 不變"""
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def make_doc_version(custom: str = "") -> str:
    """
    版號：優先使用呼叫方傳入的自訂字串，
    否則自動產生 ISO UTC 時間戳，例如 20240516T103045Z
    """
    if custom.strip():
        return custom.strip()
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Milvus primary key 格式：{doc_id}_{chunk_index:04d}"""
    return f"{doc_id}_{chunk_index:04d}"


# ── Token 估算 ────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
    """
    輕量估算，不呼叫任何 tokenizer：
    中文字 ≈ 1 token；英文單字 ≈ 1 token
    """
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    english = len(re.findall(r"[a-zA-Z0-9]+", text))
    return chinese + english


# ── 檔案類型判斷 ──────────────────────────────────────────

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}

def get_file_type(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    mapping = {
        ".pdf" : "pdf",
        ".txt" : "txt",
        ".docx": "docx",
        ".xlsx": "xlsx",
        ".xls" : "xlsx",
    }
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return mapping.get(suffix, "unknown")
