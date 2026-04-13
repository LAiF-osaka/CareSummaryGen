"""E2E テスト（実 Ollama）。

Ollama サーバー稼働時のみ実行される。
@pytest.mark.ollama デコレータ付き。
conftest.py で Ollama 未接続時は自動スキップされる。
"""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLE_MEDICAL_RECORD


@pytest.mark.ollama
class TestOllamaE2E:
    """実 Ollama サーバーを使った E2E 検証。"""

    def test_graph_with_real_ollama(self):
        """実 LLM でグラフが完了すること。"""
        from graph.builder import build_nursing_summary_graph

        graph = build_nursing_summary_graph()

        initial_state = {
            "patient_id": "TEST001",
            "raw_context": SAMPLE_MEDICAL_RECORD,
            "hospital": "hanwa",
            "chunks": [],
            "chunk_index": [],
            "template_id": "hanwa",
            "template": {},
            "search_plan": [],
            "section_results": {},
            "current_section_idx": 0,
            "search_iteration": 0,
            "max_search_iterations": 2,  # 実 LLM は遅いので上限を小さくする
            "_search_results": [],
            "_section_sufficient": False,
            "draft_summary": "",
            "reflection_feedback": "",
            "reflection_approved": False,
            "iteration_count": 0,
            "max_iterations": 1,  # Reflection も1回に制限
            "final_summary": "",
            "error": None,
        }

        result = graph.invoke(initial_state)

        assert result["final_summary"], "final_summary が空です"
        assert len(result["final_summary"]) > 100
        assert result["iteration_count"] >= 1

    def test_api_with_real_ollama(self):
        """実 LLM で FastAPI エンドポイントが正常動作すること。"""
        from app import app

        with TestClient(app) as client:
            response = client.post(
                "/ask",
                json={
                    "context": SAMPLE_MEDICAL_RECORD,
                    "patient_id": "TEST001",
                    "template_id": "hanwa",
                },
                timeout=600,  # 120B モデル用の長いタイムアウト
            )

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert len(data["answer"]) > 100
