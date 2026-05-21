"""
services/model_service.py — Ollama 模型層
職責：
  1. describe_image()   — 圖片/圖表 → 結構化文字（多模態）
  2. get_embedding()    — 文字 → 向量
  3. get_embeddings()   — 批次向量化

不含任何文件解析、chunking、DB 邏輯。
"""
from __future__ import annotations
import base64
from typing import List

import requests

from app.core.config import settings
from app.utils.helpers import get_logger

logger = get_logger(__name__)

_GENERATE_URL  = f"{settings.OLLAMA_BASE_URL}/api/generate"
_EMBEDDING_URL = f"{settings.OLLAMA_BASE_URL}/api/embeddings"

# ── Vision ────────────────────────────────────────────────

_VISION_PROMPT = """你是技術文件整理專家，請仔細觀察這張圖表或圖片，以繁體中文輸出：

1. **圖表類型**：（流程圖／架構圖／示意圖／表格／截圖／照片）
2. **結構說明**：用有序清單描述節點、箭頭方向、邏輯順序
3. **關鍵元素**：列出所有方塊、節點、標籤的名稱
4. **補充說明**：圖中的文字標注或特殊說明

格式：直接輸出 Markdown，不加任何前言後記。
若圖片為空或無法辨識，輸出：（圖表內容無法辨識）
"""


def describe_image(img_bytes: bytes, context: str = "") -> str:
    """
    將圖片送給 Ollama Vision 模型，回傳結構化繁體中文描述。
    context：可傳入圖表的背景說明，幫助模型更準確理解。
    """
    b64 = base64.b64encode(img_bytes).decode()
    prompt = f"背景：{context}\n\n{_VISION_PROMPT}" if context else _VISION_PROMPT

    payload = {
        "model" : settings.VISION_MODEL,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1000},
    }
    try:
        resp = requests.post(_GENERATE_URL, json=payload, timeout=180)
        resp.raise_for_status()
        result = resp.json().get("response", "").strip()
        logger.debug("Vision 描述完成（%d chars）", len(result))
        return result
    except requests.exceptions.ConnectionError:
        msg = "⚠️ 無法連接 Ollama，請確認 `ollama serve` 已執行"
        logger.error(msg)
        return msg
    except Exception as e:
        logger.error("Vision 呼叫失敗：%s", e)
        return f"⚠️ 圖片描述失敗：{e}"


# ── Embedding ─────────────────────────────────────────────

# qwen3-embedding 建議搭配 instruction prefix 提升檢索精準度
_EMBED_DOC_PREFIX   = "Represent this document for retrieval: "
_EMBED_QUERY_PREFIX = "Represent this query for retrieval: "


def get_embedding(text: str, is_query: bool = False) -> list[float]:
    """
    單筆文字向量化，回傳 float list。
    qwen3-embedding 支援 instruction prefix，提升中文檢索精準度：
      - 文件 embed 時加 doc prefix
      - 查詢 embed 時加 query prefix
    """
    if not text.strip():
        return [0.0] * settings.VECTOR_DIM

    # 若是 qwen3-embedding 系列，加入 instruction prefix
    model = settings.EMBEDDING_MODEL
    if "qwen3" in model.lower():
        prefix = _EMBED_QUERY_PREFIX if is_query else _EMBED_DOC_PREFIX
        text   = prefix + text

    payload = {
        "model" : model,
        "prompt": text,
    }
    try:
        resp = requests.post(_EMBEDDING_URL, json=payload, timeout=60)
        resp.raise_for_status()
        vec = resp.json().get("embedding", [])
        if not vec:
            raise ValueError("Ollama 回傳空向量")
        return vec
    except requests.exceptions.ConnectionError:
        logger.error("無法連接 Ollama embedding 端點")
        raise
    except Exception as e:
        logger.error("Embedding 失敗：%s", e)
        raise


def get_embeddings(texts: List[str]) -> List[list[float]]:
    """批次向量化，回傳對應順序的向量 list"""
    results = []
    for i, text in enumerate(texts):
        logger.debug("Embedding %d/%d ...", i + 1, len(texts))
        results.append(get_embedding(text))
    return results