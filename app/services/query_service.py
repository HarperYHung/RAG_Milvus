"""
services/query_service.py — RAG Query 核心層
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List

import requests

from app.core.config import settings
from app.services.model_service import get_embedding
from app.services.vectordb_service import ensure_collection, get_client
from app.utils.helpers import get_logger

logger = get_logger(__name__)

_GENERATE_URL = f"{settings.OLLAMA_BASE_URL}/api/generate"
_CHAT_URL     = f"{settings.OLLAMA_BASE_URL}/api/chat"


@dataclass
class Citation:
    chunk_id    : str
    doc_id      : str
    doc_version : str
    source_file : str
    page        : int
    chunk_index : int
    title       : str
    content     : str
    score       : float
    rerank_score: float = 0.0


@dataclass
class QueryResult:
    question   : str
    answer     : str
    citations  : List[Citation] = field(default_factory=list)
    history    : List[dict]     = field(default_factory=list)


def retrieve(
    query_vector : list[float],
    top_k        : int = 10,
    doc_id       : str | None = None,
    doc_version  : str | None = None,
) -> List[Citation]:
    ensure_collection()
    client = get_client()
    col    = settings.MILVUS_COLLECTION

    filters = []
    if doc_id:
        filters.append(f'doc_id == "{doc_id}"')
    if doc_version:
        filters.append(f'doc_version == "{doc_version}"')
    expr = " and ".join(filters) if filters else None

    results = client.search(
        collection_name = col,
        data            = [query_vector],
        anns_field      = "vector",
        limit           = top_k,
        filter          = expr,
        output_fields   = [
            "chunk_id", "doc_id", "doc_version",
            "source_file", "page", "chunk_index",
            "title", "content",
        ],
        search_params   = {"metric_type": "IP", "params": {"ef": 64}},
    )

    citations = []
    for hit in (results[0] if results else []):
        e = hit["entity"]
        citations.append(Citation(
            chunk_id    = e.get("chunk_id", ""),
            doc_id      = e.get("doc_id", ""),
            doc_version = e.get("doc_version", ""),
            source_file = e.get("source_file", ""),
            page        = e.get("page", 0),
            chunk_index = e.get("chunk_index", 0),
            title       = e.get("title", ""),
            content     = e.get("content", ""),
            score       = float(hit.get("distance", 0)),
        ))

    logger.debug("檢索完成：%d 個候選 chunk", len(citations))
    return citations


def filter_by_threshold(
    citations : list[Citation],
    threshold : float,
) -> list[Citation]:
    if threshold <= 0:
        return citations
    filtered = [c for c in citations if c.score >= threshold]
    dropped  = len(citations) - len(filtered)
    if dropped:
        logger.debug("Threshold=%.2f 過濾掉 %d 個低相似度 chunk", threshold, dropped)
    return filtered


def rerank(
    question  : str,
    candidates: List[Citation],
    top_n     : int = 3,
    model     : str | None = None,
) -> List[Citation]:
    rerank_model = model or getattr(settings, "RERANK_MODEL", settings.VISION_MODEL)
    scored = []

    for c in candidates:
        passage = c.content[:400].replace("\n", " ")
        prompt = (
            f"<instruct>Given a web search query, retrieve relevant passages that answer the query.</instruct>\n"
            f"<query>{question}</query>\n"
            f"<document>{passage}</document>"
        )

        try:
            resp = requests.post(
                _GENERATE_URL,
                json={
                    "model"  : rerank_model,
                    "prompt" : prompt,
                    "stream" : False,
                    "options": {"temperature": 0, "num_predict": 5},
                },
                timeout=30,
            )
            raw = resp.json().get("response", "").strip().lower()
            logger.debug("Rerank chunk=%s raw=%r", c.chunk_id, raw)

            if "yes" in raw:
                score = 1.0
            elif "no" in raw:
                score = 0.1
            else:
                nums = re.findall(r"\b(10|[0-9](?:\.\d+)?)\b", raw)
                score = min(float(nums[0]) / 10.0, 1.0) if nums else max(float(c.score), 0.0)

        except Exception as e:
            logger.warning("Rerank 失敗（chunk %s）：%s", c.chunk_id, e)
            score = max(float(c.score), 0.0)

        c.rerank_score = score
        scored.append(c)

    scored.sort(key=lambda x: x.rerank_score, reverse=True)
    result = scored[:top_n]
    logger.debug("Rerank 完成：Top-%d 分數 %s",
                 top_n, [f"{c.rerank_score:.2f}" for c in result])
    return result


_SYSTEM_PROMPT = """你是一位專業知識庫助理，使用繁體中文回答。

以下是從知識庫檢索到的參考文件，請根據這些內容回答問題：

{context}

回答要求：
1. 根據上方參考文件的內容來回答
2. 回答中標注來源，格式：（來源：檔名，第 X 頁）
3. 回答結尾列出「📚 參考來源」，包含用到的檔名與頁碼
4. 使用繁體中文回答"""

_CONTEXT_TEMPLATE = """---
【文件 {idx}】{source_file}｜第 {page} 頁｜{title}
{content}
---"""


def _build_context(citations: List[Citation]) -> str:
    parts = []
    for i, c in enumerate(citations, start=1):
        parts.append(_CONTEXT_TEMPLATE.format(
            idx         = i,
            source_file = c.source_file,
            page        = c.page if c.page else "N/A",
            title       = c.title or f"第 {c.chunk_index + 1} 段",
            content     = c.content[:800],
        ))
    return "\n".join(parts)


def generate(
    question      : str,
    citations     : List[Citation],
    history       : List[dict],
    model         : str | None = None,
    temperature   : float = 0.3,
    max_tokens    : int   = 1500,
    system_prompt : str   = "",
) -> str:
    llm     = model or settings.VISION_MODEL
    context = _build_context(citations)

    if system_prompt.strip():
        system = system_prompt.strip() + "\n\n參考文件：\n" + context
    else:
        system = _SYSTEM_PROMPT.format(context=context)

    messages = [{"role": "system", "content": system}]
    messages.extend(history[-10:])
    messages.append({"role": "user", "content": question})

    try:
        resp = requests.post(
            _CHAT_URL,
            json={
                "model"   : llm,
                "messages": messages,
                "stream"  : False,
                "think"   : False,
                "options" : {
                    "temperature" : temperature,
                    "num_predict" : max_tokens,
                    "num_ctx"     : 8192,
                },
            },
            timeout=120,
        )
        resp.raise_for_status()
        data    = resp.json()
        message = data.get("message", {})
        answer  = message.get("content", "").strip()

        if not answer:
            logger.warning("LLM 回傳空回答，完整回應：%s", data)
            answer = "⚠️ 模型回傳空回答，請重試。"

        return answer

    except requests.exceptions.Timeout:
        logger.error("LLM 生成逾時")
        return "⚠️ 模型回應逾時，請縮小 Max Tokens 或稍後重試。"
    except Exception as e:
        logger.error("LLM 生成失敗：%s", e)
        return f"⚠️ 生成回答時發生錯誤：{e}"


def query(
    question            : str,
    history             : List[dict] | None = None,
    top_k               : int   = 10,
    top_n               : int   = 3,
    similarity_threshold: float = 0.0,
    use_rerank          : bool  = False,
    temperature         : float = 0.3,
    max_tokens          : int   = 1500,
    system_prompt       : str   = "",
    doc_id              : str | None = None,
    doc_version         : str | None = None,
) -> QueryResult:
    history = history or []

    logger.info(
        "Query 開始：question='%s'  top_k=%d  threshold=%.2f  rerank=%s  temp=%.1f",
        question, top_k, similarity_threshold, use_rerank, temperature,
    )

    try:
        vec = get_embedding(question, is_query=True)
    except Exception as e:
        logger.error("Embedding 失敗：%s", e)
        return QueryResult(
            question = question,
            answer   = f"⚠️ Embedding 失敗：{e}（請確認 Ollama 已啟動）",
            citations= [], history=history,
        )

    candidates = retrieve(vec, top_k=top_k, doc_id=doc_id, doc_version=doc_version)
    candidates = filter_by_threshold(candidates, similarity_threshold)

    if not candidates:
        answer = "根據現有文件，我找不到此問題的相關資訊。"
        return QueryResult(question=question, answer=answer,
                           citations=[], history=history)

    if use_rerank and len(candidates) > top_n:
        final_citations = rerank(question, candidates, top_n=top_n)
    else:
        final_citations = candidates[:top_n]

    answer = generate(
        question      = question,
        citations     = final_citations,
        history       = history,
        temperature   = temperature,
        max_tokens    = max_tokens,
        system_prompt = system_prompt,
    )

    updated_history = history + [
        {"role": "user",      "content": question},
        {"role": "assistant", "content": answer},
    ]

    logger.info("Query 完成：citations=%d", len(final_citations))
    return QueryResult(
        question = question,
        answer   = answer,
        citations= final_citations,
        history  = updated_history,
    )