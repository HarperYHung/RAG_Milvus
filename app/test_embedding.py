"""
embedding_test.py — 產生向量供 Attu Vector Search 使用
執行方式：python -m app.embedding_test
"""
import requests
from app.core.config import settings

def get_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{settings.OLLAMA_BASE_URL}/api/embed",
        json={
            "model": settings.EMBED_MODEL,
            "input": text,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"][0]


if __name__ == "__main__":
    text = input("輸入測試文字：")
    vector = get_embedding(text)
    print(f"\n完整向量（複製貼到 Attu）：\n{vector}")