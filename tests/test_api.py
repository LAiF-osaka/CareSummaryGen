"""FastAPI エンドポイントのテスト（LLM不要な部分のみ）。"""

import pytest
from fastapi.testclient import TestClient

from app import app


@pytest.fixture
def client():
    """FastAPI テストクライアント。"""
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """ヘルスチェックが正常応答すること。"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "model" in data


def test_templates_list(client):
    """テンプレート一覧が取得できること。"""
    response = client.get("/templates")
    assert response.status_code == 200
    templates = response.json()
    assert len(templates) >= 2
    ids = [t["id"] for t in templates]
    assert "hanwa" in ids


def test_ask_empty_context(client):
    """空の context で 422 が返ること。"""
    response = client.post("/ask", json={"context": ""})
    assert response.status_code == 422


def test_ask_missing_context(client):
    """context なしで 422 が返ること。"""
    response = client.post("/ask", json={})
    assert response.status_code == 422


def test_ask_invalid_template(client):
    """存在しないテンプレートで 400 が返ること。"""
    response = client.post(
        "/ask",
        json={"context": "テスト記録", "template_id": "nonexistent"},
    )
    assert response.status_code == 400
