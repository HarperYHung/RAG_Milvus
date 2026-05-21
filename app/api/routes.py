"""
api/routes.py — FastAPI 路由層
職責：HTTP 請求/回應的轉換，不含任何業務邏輯
"""
from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.models.schemas import (
    ChunkStrategy, IngestRequest, IngestResponse, UpsertResult,
    QueryRequest, QueryResponse, CitationOut,
)
from app.services import ingest_service, vectordb_service
from app.utils.helpers import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/api/v1")


# ── Ingest ────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse, summary="上傳並處理文件")
async def ingest(
    file           : UploadFile = File(..., description="PDF / TXT / DOCX / XLSX / 圖片"),
    chunk_strategy : str        = Form("size",  description="header 或 size"),
    chunk_size     : int        = Form(512,     description="每塊最大 token 數（size 切法）"),
    chunk_overlap  : int        = Form(50,      description="重疊 token 數（size 切法）"),
    doc_version    : str        = Form("",      description="版號，空白則自動填入時間戳"),
):
    try:
        strategy = ChunkStrategy(chunk_strategy)
    except ValueError:
        raise HTTPException(400, f"chunk_strategy 只接受 'header' 或 'size'，收到：{chunk_strategy}")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(400, "上傳檔案為空")

    request = IngestRequest(
        chunk_strategy = strategy,
        chunk_size     = chunk_size,
        chunk_overlap  = chunk_overlap,
        doc_version    = doc_version,
    )

    response = ingest_service.run_ingest(
        file_bytes = file_bytes,
        filename   = file.filename or "unknown",
        request    = request,
    )
    return response


# ── VectorDB ──────────────────────────────────────────────

@router.get("/vectordb/stats", summary="查詢 Collection 統計")
async def vectordb_stats():
    try:
        return vectordb_service.collection_stats()
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/vectordb/doc/{doc_id}", summary="刪除文件的所有向量")
async def delete_doc(doc_id: str, doc_version: str | None = None):
    try:
        count = vectordb_service.delete_by_doc(doc_id, doc_version)
        return {"deleted": count, "doc_id": doc_id, "doc_version": doc_version}
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Query ─────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, summary="RAG 問答（含 Citation）")
async def query(req: QueryRequest):
    from app.services.query_service import query as run_query
    try:
        result = run_query(
            question             = req.question,
            history              = req.history,
            top_k                = req.top_k,
            top_n                = req.top_n,
            similarity_threshold = req.similarity_threshold,
            use_rerank           = req.use_rerank,
            temperature          = req.temperature,
            max_tokens           = req.max_tokens,
            system_prompt        = req.system_prompt,
            doc_id               = req.doc_id,
            doc_version          = req.doc_version,
        )
        citations_out = [
            CitationOut(
                chunk_id     = c.chunk_id,
                source_file  = c.source_file,
                page         = c.page,
                chunk_index  = c.chunk_index,
                title        = c.title,
                content      = c.content[:300],
                score        = round(c.score, 4),
                rerank_score = round(c.rerank_score, 4),
            )
            for c in result.citations
        ]
        return QueryResponse(
            question  = result.question,
            answer    = result.answer,
            citations = citations_out,
            history   = result.history,
        )
    except Exception as e:
        raise HTTPException(500, str(e))


# ── Health ────────────────────────────────────────────────

@router.get("/health", summary="健康檢查")
async def health():
    return {"status": "ok"}