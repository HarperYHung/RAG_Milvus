"""
utils/helpers.py — 純工具函式，不依賴任何業務層
"""
import hashlib
import logging
import re
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
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        fh = logging.FileHandler(settings.LOG_DIR / "rag.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ── 文件唯一碼 ────────────────────────────────────────────

def compute_doc_id(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes).hexdigest()[:16]


def make_doc_version(custom: str = "") -> str:
    return custom.strip() if custom.strip() else "V1"


def make_chunk_id(doc_id: str, chunk_index: int) -> str:
    return f"{doc_id}_{chunk_index:04d}"


# ── Token 估算 ────────────────────────────────────────────

def estimate_tokens(text: str) -> int:
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
        ".doc" : "docx",
        ".xlsx": "xlsx",
        ".xls" : "xlsx",
        ".pptx": "pptx",
        ".ppt" : "pptx",
    }
    if suffix in IMAGE_SUFFIXES:
        return "image"
    return mapping.get(suffix, "unknown")