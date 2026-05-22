"""
services/vectordb_service.py — Milvus 向量資料庫層
職責：
  1. 建立 / 確保 Collection 存在（含 schema 定義）
  2. upsert()   — 依 doc_id + chunk_id 去重後寫入（舊版本不刪除，依 doc_version 區分）
  3. delete_by_doc() — 刪除指定 doc_id 的所有向量（整份文件重處理時用）
  4. search()   — 向量相似度查詢（留給未來 Query 模組使用）

Schema 設計（Milvus）：
  chunk_id    [VARCHAR 64]  PK  — "{doc_id}_{chunk_index:04d}"
  doc_id      [VARCHAR 16]      — SHA-256[:16]，文件唯一碼（內容 hash）
  doc_version [VARCHAR 32]      — 版號（時間戳或自訂），支援多版本共存
  source_file [VARCHAR 256]     — 原始檔名
  file_type   [VARCHAR 16]      — pdf / txt / docx / xlsx / image
  page        [INT64]           — 頁碼
  chunk_index [INT64]           — 文件內第幾個 chunk
  title       [VARCHAR 512]     — 標題（header 切法才有）
  strategy    [VARCHAR 16]      — header / size
  content     [VARCHAR 65535]   — chunk 原文
  vector      [FLOAT_VECTOR]    — embedding 向量
"""
from __future__ import annotations
from typing import List

from pymilvus import (
    MilvusClient,
    CollectionSchema,
    FieldSchema,
    DataType,
    MilvusException,
)

from app.core.config import settings
from app.models.schemas import EmbeddedChunk, UpsertResult
from app.utils.helpers import get_logger

logger = get_logger(__name__)

# ── Milvus Client（單例）─────────────────────────────────

_client: MilvusClient | None = None

def get_client() -> MilvusClient:
    global _client
    if _client is None:
        uri = f"http://{settings.MILVUS_HOST}:{settings.MILVUS_PORT}"
        _client = MilvusClient(uri=uri)
        logger.info("Milvus 連線：%s", uri)
    return _client


# ── Collection Schema ─────────────────────────────────────

def _build_schema() -> CollectionSchema:
    fields = [
        FieldSchema("chunk_id",    DataType.VARCHAR, max_length=64,    is_primary=True),
        FieldSchema("doc_id",      DataType.VARCHAR, max_length=16),
        FieldSchema("doc_version", DataType.VARCHAR, max_length=32),
        FieldSchema("source_file", DataType.VARCHAR, max_length=256),
        FieldSchema("file_type",   DataType.VARCHAR, max_length=16),
        FieldSchema("page",        DataType.INT64),
        FieldSchema("chunk_index", DataType.INT64),
        FieldSchema("title",       DataType.VARCHAR, max_length=512),
        FieldSchema("strategy",    DataType.VARCHAR, max_length=16),
        FieldSchema("content",     DataType.VARCHAR, max_length=65535),
        FieldSchema("vector",      DataType.FLOAT_VECTOR, dim=settings.VECTOR_DIM),
    ]
    return CollectionSchema(fields=fields, enable_dynamic_field=True)


def _create_collection_with_index(client: MilvusClient, col: str) -> None:
    """
    pymilvus 3.x：透過 create_collection 的 index_params 參數
    一次完成建立 + 索引，避免非同步問題。
    """
    # 準備索引參數
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name  = "vector",
        index_type  = "HNSW",
        metric_type = "IP",
        params      = {"M": 16, "efConstruction": 200},
    )
    client.create_collection(
        collection_name = col,
        schema          = _build_schema(),
        index_params    = index_params,
    )
    logger.info("Collection '%s' 建立完成（含索引）", col)


def ensure_collection() -> None:
    """確保 Collection 存在、有索引、且已載入記憶體"""
    client = get_client()
    col    = settings.MILVUS_COLLECTION

    # 不存在 → 全新建立
    if not client.has_collection(col):
        logger.info("建立 Collection '%s'（dim=%d）", col, settings.VECTOR_DIM)
        _create_collection_with_index(client, col)

    # load：若失敗（例如缺索引）→ 刪除重建再 load
    try:
        client.load_collection(col)
        logger.debug("Collection '%s' 已載入", col)
    except Exception as e:
        logger.warning("load_collection 失敗（%s），刪除並重建...", e)
        client.drop_collection(col)
        _create_collection_with_index(client, col)
        client.load_collection(col)
        logger.info("Collection '%s' 重建並載入完成", col)


# ── Upsert ────────────────────────────────────────────────

def upsert(embedded_chunks: List[EmbeddedChunk]) -> UpsertResult:
    """
    Upsert 邏輯：
      - chunk_id 為 PK，相同 chunk_id 寫入時 Milvus 自動覆蓋（upsert）
      - 同一 doc_id 不同 doc_version → chunk_id 不同（因為 doc_id 不同）
        → 多版本可共存，查詢時可依 doc_version filter
      - 同一檔案重新上傳但內容相同 → doc_id 相同 → chunk_id 相同 → 自動覆蓋
    """
    if not embedded_chunks:
        return UpsertResult(doc_id="", doc_version="",
                            upserted_count=0, message="無資料")

    ensure_collection()
    client  = get_client()
    col     = settings.MILVUS_COLLECTION

    # 組成 Milvus 期望的行格式
    rows = []
    for ec in embedded_chunks:
        m = ec.chunk.meta
        rows.append({
            "chunk_id"   : ec.chunk.chunk_id,
            "doc_id"     : m.doc_id,
            "doc_version": m.doc_version,
            "source_file": m.source_file,
            "file_type"  : m.file_type.value,
            "page"       : m.page,
            "chunk_index": m.chunk_index,
            "title"      : m.title,
            "strategy"   : m.chunk_strategy.value,
            "content"    : ec.chunk.content[:65530],  # Milvus VARCHAR 上限
            "vector"     : ec.vector,
        })

    doc_id      = embedded_chunks[0].chunk.meta.doc_id
    doc_version = embedded_chunks[0].chunk.meta.doc_version

    try:
        result = client.upsert(collection_name=col, data=rows)
        upserted = result.get("upsert_count", len(rows))
        logger.info("Upsert 完成：doc_id=%s  version=%s  count=%d",
                    doc_id, doc_version, upserted)
        return UpsertResult(
            doc_id         = doc_id,
            doc_version    = doc_version,
            upserted_count = upserted,
            message        = "OK",
        )
    except MilvusException as e:
        logger.error("Upsert 失敗：%s", e)
        raise


# ── Delete ────────────────────────────────────────────────

def delete_by_doc(doc_id: str, doc_version: str | None = None) -> int:
    """
    刪除指定文件的所有向量。
    doc_version=None → 刪除該 doc_id 所有版本。
    回傳刪除數量。
    """
    ensure_collection()
    client = get_client()
    col    = settings.MILVUS_COLLECTION

    if doc_version:
        expr = f'doc_id == "{doc_id}" and doc_version == "{doc_version}"'
    else:
        expr = f'doc_id == "{doc_id}"'

    result = client.delete(collection_name=col, filter=expr)
    deleted = result.get("delete_count", 0)
    logger.info("刪除：doc_id=%s  version=%s  count=%d",
                doc_id, doc_version, deleted)
    return deleted


# ── Search ────────────────────────────────────────────────

def search(
    query_vector : list[float],
    top_k        : int = 5,
    doc_id       : str | None = None,
    doc_version  : str | None = None,
) -> list[dict]:
    """
    向量相似度搜尋。
    可選擇性過濾 doc_id 或 doc_version。
    回傳 list of dict（含 content、meta、score）。
    """
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
        output_fields   = ["chunk_id", "doc_id", "doc_version",
                           "source_file", "page", "title",
                           "strategy", "content"],
    )

    hits = []
    for hit in (results[0] if results else []):
        hits.append({
            "score"      : hit["distance"],
            "chunk_id"   : hit["entity"]["chunk_id"],
            "doc_id"     : hit["entity"]["doc_id"],
            "doc_version": hit["entity"]["doc_version"],
            "source_file": hit["entity"]["source_file"],
            "page"       : hit["entity"]["page"],
            "title"      : hit["entity"]["title"],
            "content"    : hit["entity"]["content"],
        })
    return hits


# ── Stats ─────────────────────────────────────────────────

def collection_stats() -> dict:
    """回傳 Collection 基本統計，供 UI 顯示"""
    ensure_collection()
    client = get_client()
    col    = settings.MILVUS_COLLECTION
    stats  = client.get_collection_stats(col)
    total  = int(stats.get("row_count", 0))
    return {
        "collection"  : col,
        "total_chunks": total,
        "dim"         : settings.VECTOR_DIM,
    }