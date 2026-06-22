"""E2E テスト（v2・LLM モック付き）。

Ollama 不要で、v2 グラフ（ingest → single_pass/fanout → section_worker →
assemble → consistency → finalize）の全フローを検証する。
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLE_MEDICAL_RECORD


def _mock_chat_json_single_pass(prompt, schema, **kwargs):
    """single_pass / section_worker / consistency の chat_json モック。

    プロンプト内容で呼び出し元を判別し、適切な構造化出力を返す。
    """
    if "need_more" in str(schema):  # section_worker 補完検索
        # gap 分析済み（missing_points 明示）で充足停止する準拠モデルを模す
        return {"need_more": False, "missing_points": []}
    if "claims" in str(schema):  # consistency
        return {"claims": ["2024年1月15日 BP 138/82 mmHg"]}
    if "sections" in str(schema):  # single_pass
        keys = [
            "instruction",
            "medical_equipment",
            "nursing_process",
            "patient_condition",
            "risks",
            "others",
        ]
        return {
            "sections": [
                {
                    "section_key": k,
                    "body": f"{k}の本文。2024年1月15日 BP 138/82 mmHg。",
                    "cited_dates": ["20240115"],
                }
                for k in keys
            ]
        }
    # section_worker（単一セクション）
    return {
        "reasoning": "根拠",
        "body": "セクション本文。2024年1月15日 BP 138/82 mmHg。",
        "cited_dates": ["20240115"],
    }


@pytest.fixture
def mock_llm():
    """v2 ノードの chat_json をモックする。"""
    with (
        patch(
            "graph.nodes.single_pass.chat_json",
            side_effect=_mock_chat_json_single_pass,
        ),
        patch(
            "graph.nodes.section_worker.chat_json",
            side_effect=_mock_chat_json_single_pass,
        ),
        patch(
            "graph.nodes.consistency.chat_json",
            side_effect=_mock_chat_json_single_pass,
        ),
    ):
        yield


def _initial_state(template_id="hanwa", context=SAMPLE_MEDICAL_RECORD):
    return {
        "patient_id": "TEST001",
        "raw_context": context,
        "hospital": "hanwa",
        "template_id": template_id,
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


class TestGraphE2E:
    """v2 グラフのフロー検証。"""

    def test_single_pass_path(self, mock_llm):
        """短い入力で single_pass 経路を通り全セクションが生成されること。"""
        from graph.builder import build_nursing_summary_graph

        graph = build_nursing_summary_graph()
        result = graph.invoke(_initial_state())

        assert result["final_summary"]
        assert "指導した内容" in result["final_summary"]
        assert "継続される問題" in result["final_summary"]
        # 全セクションに本文が入る
        assert len(result["section_results"]) == 6
        for r in result["section_results"].values():
            assert r["body"]

    def test_ingest_builds_grep_index(self, mock_llm):
        """ingest が日付チャンクと grep_index を構築すること。"""
        from graph.builder import build_nursing_summary_graph

        graph = build_nursing_summary_graph()
        result = graph.invoke(_initial_state())
        dates = [c["date"] for c in result["grep_index"] if c.get("date")]
        assert "20230209" in dates

    def test_shinkinen_template(self, mock_llm):
        """新記念テンプレート（2セクション）でも完了すること。"""
        from graph.builder import build_nursing_summary_graph

        graph = build_nursing_summary_graph()
        result = graph.invoke(_initial_state(template_id="shinkinen"))
        assert result["final_summary"]
        assert len(result["section_results"]) == 2


class TestAPIE2E:
    """FastAPI 経由の v2 検証。"""

    def test_ask_endpoint(self, mock_llm):
        """POST /ask が 200 でサマリーと review_flags を返すこと。"""
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
        data = response.json()
        assert data["answer"]
        assert data["template_id"] == "hanwa"
        assert "review_flags" in data
