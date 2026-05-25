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
_CHAT_URL      = f"{settings.OLLAMA_BASE_URL}/api/chat"
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


def _extract_response_text(data: dict) -> str:
    """
    從 Ollama API 回傳的 JSON 中安全地取出文字內容。
    相容：
      /api/generate → response 欄位
      /api/chat     → message.content 欄位
      thinking 模式 → content 為空時，從 message.thinking 後的串流補撈
                      或嘗試頂層 content 欄位
    """
    # /api/generate 格式
    if "response" in data:
        text = str(data["response"]).strip()
        if text:
            return text

    # /api/chat 格式
    if "message" in data:
        msg = data["message"]
        content = str(msg.get("content", "")).strip()
        if content:
            return content
        # gemma4 thinking 模式：content 為空時，
        # thinking 欄位包含思考過程，實際回答需要另外觸發
        # → 直接取 thinking 末段作為備援（比空字串有用）
        thinking = str(msg.get("thinking", "")).strip()
        if thinking:
            logger.debug("Vision content 為空，使用 thinking 內容作為備援（%d chars）",
                         len(thinking))
            return thinking

    # 頂層 content
    if "content" in data:
        text = str(data["content"]).strip()
        if text:
            return text

    logger.warning("Vision API 回傳格式未知，完整 keys：%s", list(data.keys()))
    return ""


def describe_image(img_bytes: bytes, context: str = "") -> str:
    """
    將圖片送給 Ollama Vision 模型，回傳結構化繁體中文描述。
    context：可傳入圖表的背景說明，幫助模型更準確理解。

    先嘗試 /api/generate，若回傳空字串再改用 /api/chat，
    確保相容不同版本的 Ollama 和不同模型的回傳格式。
    """
    if not img_bytes:
        logger.warning("describe_image 收到空 bytes，跳過")
        return ""

    b64    = base64.b64encode(img_bytes).decode()
    prompt = f"背景：{context}\n\n{_VISION_PROMPT}" if context else _VISION_PROMPT

    # 方式一：/api/generate（Ollama 標準 Vision API）
    payload_generate = {
        "model" : settings.VISION_MODEL,
        "prompt": prompt,
        "images": [b64],
        "stream": False,
        "think" : False,   # 關閉 thinking 模式，確保回答在 response 欄位
        "options": {"temperature": 0.1, "num_predict": 1000},
    }
    try:
        resp = requests.post(_GENERATE_URL, json=payload_generate, timeout=180)
        resp.raise_for_status()
        data   = resp.json()
        result = _extract_response_text(data)

        if result:
            logger.debug("Vision 描述完成（%d chars）", len(result))
            return result

        # 回傳空字串時 log 完整回應供 debug
        logger.warning(
            "Vision /api/generate 回傳空字串，完整回應：%s",
            str(data)[:300],
        )
    except requests.exceptions.ConnectionError:
        msg = "⚠️ 無法連接 Ollama，請確認 `ollama serve` 已執行"
        logger.error(msg)
        return msg
    except Exception as e:
        logger.warning("Vision /api/generate 失敗：%s，改用 /api/chat", e)

    # 方式二：/api/chat（部分模型需要用 chat 格式才能處理圖片）
    payload_chat = {
        "model" : settings.VISION_MODEL,
        "stream": False,
        "think" : False,   # 關閉 thinking 模式
        "messages": [
            {
                "role"   : "user",
                "content": prompt,
                "images" : [b64],
            }
        ],
        "options": {"temperature": 0.1, "num_predict": 1000},
    }
    try:
        resp = requests.post(_CHAT_URL, json=payload_chat, timeout=180)
        resp.raise_for_status()
        data   = resp.json()
        result = _extract_response_text(data)

        if result:
            logger.debug("Vision /api/chat 描述完成（%d chars）", len(result))
            return result

        logger.warning(
            "Vision /api/chat 也回傳空字串，完整回應：%s",
            str(data)[:300],
        )
        return "（圖表內容無法辨識）"
    except Exception as e:
        logger.error("Vision /api/chat 也失敗：%s", e)
        return f"⚠️ 圖片描述失敗：{e}"


# ── Embedding ─────────────────────────────────────────────

_EMBED_DOC_PREFIX   = "Represent this document for retrieval: "
_EMBED_QUERY_PREFIX = "Represent this query for retrieval: "


def get_embedding(text: str, is_query: bool = False) -> list[float]:
    """
    單筆文字向量化，回傳 float list。
    qwen3-embedding 支援 instruction prefix，提升中文檢索精準度。
    """
    if not text.strip():
        return [0.0] * settings.VECTOR_DIM

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