# RAG Pipeline

端對端 RAG 文件匯入系統（本地多模態版本）

## 架構

```
rag_project/
├── app/
│   ├── api/
│   │   └── routes.py          # FastAPI 路由（HTTP 層）
│   ├── core/
│   │   └── config.py          # 全域設定
│   ├── models/
│   │   └── schemas.py         # Pydantic 資料結構
│   ├── services/
│   │   ├── model_service.py   # Ollama：Vision + Embedding
│   │   ├── parser_service.py  # 文件解析：PDF/TXT/DOCX/XLSX/圖片
│   │   ├── chunking_service.py# 切割：header / size
│   │   ├── vectordb_service.py# Milvus：upsert / delete / search
│   │   └── ingest_service.py  # Pipeline 編排
│   ├── utils/
│   │   └── helpers.py         # 工具：hash / token / logger
│   └── main.py                # FastAPI App 入口
├── frontend/
│   └── index.html             # 單頁前端介面
├── uploads/                   # 暫存上傳檔案
├── logs/
│   └── rag.log
└── requirements.txt
```

## 職責分離

| 層 | 檔案 | 職責 |
|---|------|------|
| API | `routes.py` | HTTP 轉換，不含業務邏輯 |
| 編排 | `ingest_service.py` | 串接各 service，不含實作細節 |
| 解析 | `parser_service.py` | 檔案 → ParsedPage（含圖表 Vision） |
| 切割 | `chunking_service.py` | ParsedPage → Chunk list |
| 模型 | `model_service.py` | Ollama API（Vision + Embedding） |
| 向量DB | `vectordb_service.py` | Milvus CRUD |
| 資料 | `schemas.py` | 跨層資料合約 |
| 工具 | `helpers.py` | 純工具函式 |

## 前置條件

```bash
# 1. Milvus
docker compose up -d   # 使用前一份 docker-compose.yml

# 2. Ollama
ollama pull gemma3:12b        # Vision（圖表描述）
ollama pull nomic-embed-text  # Embedding
ollama serve

# 3. Python 套件
pip install -r requirements.txt
```

## 啟動

```bash
cd rag_project
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

打開瀏覽器：http://localhost:8000

API 文件：http://localhost:8000/docs

## Upsert 策略

- `doc_id` = SHA-256(file_bytes)[:16]：**內容 hash**，同一份文件重複上傳自動覆蓋
- `doc_version` = 時間戳或自訂版號：**可多版本共存**，查詢時以 `doc_version` filter
- `chunk_id` = `{doc_id}_{chunk_index:04d}`：Milvus Primary Key，upsert 時自動去重

## 環境變數（.env 選填）

```env
OLLAMA_BASE_URL=http://localhost:11434
VISION_MODEL=gemma3:12b
EMBEDDING_MODEL=nomic-embed-text
MILVUS_HOST=localhost
MILVUS_PORT=19530
VECTOR_DIM=768
```

## 支援格式

| 格式 | 文字 | 表格 | 圖片/圖表 |
|------|------|------|-----------|
| PDF  | ✅   | ✅ Markdown 表格 | ✅ 自動偵測 + Vision |
| DOCX | ✅   | ✅ | ✅ 內嵌圖片 |
| XLSX | ✅   | ✅ | — |
| TXT  | ✅   | — | — |
| 圖片 | —    | — | ✅ 直接 Vision |
