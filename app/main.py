"""
app/main.py — FastAPI 應用程式入口
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.utils.helpers import get_logger

logger = get_logger("main")

app = FastAPI(
    title   = settings.PROJECT_NAME,
    version = "1.0.0",
    docs_url= "/docs",
)

# CORS（開發環境全開）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 路由
app.include_router(router)

# 靜態前端（把 frontend/ 目錄掛載到 /）
from pathlib import Path
frontend_dir = Path(__file__).resolve().parents[1] / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

@app.on_event("startup")
async def startup():
    logger.info("RAG Pipeline 啟動（port %d）", settings.API_PORT)
