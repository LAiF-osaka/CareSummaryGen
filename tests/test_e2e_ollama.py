"""E2E テスト（v2・実 Ollama）。

Ollama 稼働時のみ実行（@pytest.mark.ollama、conftest で自動スキップ）。
"""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLE_MEDICAL_RECORD


def _initial_state():
    return {
        "patient_id": "TEST001",
        "raw_context": SAMPLE_MEDICAL_RECORD,
        "hospital": "hanwa",
        "template_id": "hanwa",
        "template": {},
        "routing": {},
        "summary_header": "",
        "chunks": [],
        "grep_index": [],
        "total_tokens": 0,
        "section_results": {},
        "draft_summary": "",
        "final_summary": "",
        "review_flags": [],
        "error": None,
    }


@pytest.mark.ollama
class TestOllamaE2E:
    """実 Ollama を使った v2 E2E。"""

    def test_graph_with_real_ollama(self):
        """実 LLM でグラフが完了し全セクションが生成されること。"""
        from graph.builder import build_nursing_summary_graph

        graph = build_nursing_summary_graph()
        result = graph.invoke(_initial_state())

        assert result["final_summary"]
        assert len(result["final_summary"]) > 100
        assert len(result["section_results"]) == 6

    def test_api_with_real_ollama(self):
        """実 LLM で /ask が 200 を返すこと。"""
        from app import app

        with TestClient(app) as client:
            response = client.post(
                "/ask",
                json={
                    "context": SAMPLE_MEDICAL_RECORD,
                    "patient_id": "TEST001",
                    "template_id": "hanwa",
                },
            )
        assert response.status_code == 200
        assert response.json()["answer"]
