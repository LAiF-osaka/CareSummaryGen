"""E2E テスト（LLM モック付き）。

Ollama サーバー不要で、グラフ全体のフローを検証する。
LLM 呼び出しをモックして各ノードの連携と状態遷移を確認する。
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from tests.conftest import SAMPLE_MEDICAL_RECORD


# --- LLM モックレスポンス ---

def _mock_chat(prompt, *, system="", format_schema=None, temperature=None):
    """llm.client.chat のモック。

    プロンプト内容に応じて適切なモックレスポンスを返す。
    """
    prompt_lower = prompt.lower() if prompt else ""

    # plan ノード: 検索クエリを JSON で返す
    if format_schema and "queries" in str(format_schema):
        return json.dumps({
            "queries": ["バイタル", "手術", "リハビリ", "退院"]
        })

    # evaluate ノード: 常に SUFFICIENT を返す（ループを1回で抜ける）
    if "SUFFICIENT" in prompt:
        return "SUFFICIENT"

    # extract ノード: セクションに応じたモック抽出結果
    if "抽出してください" in prompt:
        return "2023年2月9日 入院。BP 138/82 mmHg。右大腿骨頸部骨折。手術施行。"

    # synthesize ノード: テンプレート形式のドラフト
    if "統合し" in prompt:
        return """--- 指導した内容 ---
退院後の生活指導、転倒予防、介護方法の指導を実施。

--- 医療機器装着・挿入・処置部位 ---
JP ドレーン（術後留置、2月12日抜去）、バルーンカテーテル（2月12日抜去）

--- 入院中の看護の経過（生活状況） ---
2023年2月9日 右大腿骨頸部骨折にて入院。BP 138/82 mmHg。
2023年2月10日 人工骨頭置換術施行。術後疼痛管理。
2023年2月12日 リハビリ開始。車椅子移乗訓練。
2023年2月15日 歩行器での歩行訓練へ移行。ADL改善。
2023年2月20日 退院前カンファレンス実施。

--- 患者への病状説明及び本人・家族の受け止め方 ---
退院後生活指導を患者と家族（長女）に実施。

--- 継続される問題（今後のリスク） ---
転倒リスク、糖尿病管理、疼痛管理

--- その他 ---
退院後フォロー: 整形外科外来2週間後、訪問リハビリ週2回"""

    # reflect ノード: APPROVED を返す（ループを抜ける）
    if "品質管理" in prompt:
        return "APPROVED"

    # revise ノード（到達しないはずだがフォールバック）
    if "改善してください" in prompt:
        return prompt.split("## 元のサマリー\n")[-1].split("\n## 指摘")[0]

    # デフォルト
    return "モックレスポンス"


def _mock_chat_with_tools(prompt, tools, *, system=""):
    """llm.client.chat_with_tools のモック。

    tool calling を使わず例外を発生させ、フォールバックパスを検証する。
    """
    raise ConnectionError("Ollama not available (mock)")


@pytest.fixture
def mock_llm():
    """LLM 呼び出しをモックするフィクスチャ。"""
    with (
        patch("graph.nodes.plan.chat", side_effect=_mock_chat),
        patch("graph.nodes.search.chat", side_effect=_mock_chat),
        patch("graph.nodes.search.chat_with_tools", side_effect=_mock_chat_with_tools),
        patch("graph.nodes.extract.chat", side_effect=_mock_chat),
        patch("graph.nodes.evaluate.chat", side_effect=_mock_chat),
        patch("graph.nodes.synthesize.chat", side_effect=_mock_chat),
        patch("graph.nodes.reflect.chat", side_effect=_mock_chat),
        patch("graph.nodes.revise.chat", side_effect=_mock_chat),
    ):
        yield


# --- グラフ単体 E2E ---


class TestGraphE2E:
    """LangGraph グラフ全体のフロー検証。"""

    def test_full_graph_flow(self, mock_llm):
        """グラフが START から END まで正常に実行されること。"""
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
            "max_search_iterations": 3,
            "_search_results": [],
            "_section_sufficient": False,
            "draft_summary": "",
            "reflection_feedback": "",
            "reflection_approved": False,
            "iteration_count": 0,
            "max_iterations": 2,
            "final_summary": "",
            "error": None,
        }

        result = graph.invoke(initial_state)

        # 最終サマリーが生成されていること
        assert result["final_summary"], "final_summary が空です"
        assert len(result["final_summary"]) > 50

        # テンプレートのセクションが含まれていること
        assert "指導した内容" in result["final_summary"]
        assert "看護の経過" in result["final_summary"]

        # Reflection が実行されたこと
        assert result["iteration_count"] >= 1

        # APPROVED で早期終了したこと
        assert result["reflection_approved"] is True

    def test_all_sections_populated(self, mock_llm):
        """全セクションに抽出結果が入っていること。"""
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
            "max_search_iterations": 3,
            "_search_results": [],
            "_section_sufficient": False,
            "draft_summary": "",
            "reflection_feedback": "",
            "reflection_approved": False,
            "iteration_count": 0,
            "max_iterations": 2,
            "final_summary": "",
            "error": None,
        }

        result = graph.invoke(initial_state)

        # 阪和病院テンプレートは6セクション
        expected_keys = [
            "instruction", "medical_equipment", "nursing_process",
            "patient_condition", "risks", "others",
        ]
        for key in expected_keys:
            assert key in result["section_results"], (
                f"セクション '{key}' が section_results にありません"
            )
            assert result["section_results"][key], (
                f"セクション '{key}' の内容が空です"
            )

    def test_shinkinen_template(self, mock_llm):
        """新記念病院テンプレートでもグラフが完了すること。"""
        from graph.builder import build_nursing_summary_graph

        graph = build_nursing_summary_graph()

        initial_state = {
            "patient_id": "TEST001",
            "raw_context": SAMPLE_MEDICAL_RECORD,
            "hospital": "shinkinen",
            "chunks": [],
            "chunk_index": [],
            "template_id": "shinkinen",
            "template": {},
            "search_plan": [],
            "section_results": {},
            "current_section_idx": 0,
            "search_iteration": 0,
            "max_search_iterations": 3,
            "_search_results": [],
            "_section_sufficient": False,
            "draft_summary": "",
            "reflection_feedback": "",
            "reflection_approved": False,
            "iteration_count": 0,
            "max_iterations": 2,
            "final_summary": "",
            "error": None,
        }

        result = graph.invoke(initial_state)

        assert result["final_summary"]
        # 新記念病院は2セクション
        assert "progress" in result["section_results"]
        assert "remarks" in result["section_results"]

    def test_chunking_produces_dates(self, mock_llm):
        """日付単位のチャンク分割が正しく行われること。"""
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
            "max_search_iterations": 3,
            "_search_results": [],
            "_section_sufficient": False,
            "draft_summary": "",
            "reflection_feedback": "",
            "reflection_approved": False,
            "iteration_count": 0,
            "max_iterations": 2,
            "final_summary": "",
            "error": None,
        }

        result = graph.invoke(initial_state)

        # 5日分の日付チャンクが生成されること
        assert len(result["chunks"]) == 5
        dates = [c["date"] for c in result["chunk_index"]]
        assert "20230209" in dates
        assert "20230220" in dates


# --- FastAPI E2E ---


class TestAPIE2E:
    """FastAPI エンドポイント経由の E2E 検証。"""

    def test_ask_endpoint_full_flow(self, mock_llm):
        """POST /ask が正常にサマリーを返すこと。"""
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
        assert "answer" in data
        assert len(data["answer"]) > 50
        assert data["template_id"] == "hanwa"
        assert data["iteration_count"] >= 1

    def test_ask_endpoint_shinkinen(self, mock_llm):
        """新記念病院テンプレートでも API が正常動作すること。"""
        from app import app

        with TestClient(app) as client:
            response = client.post(
                "/ask",
                json={
                    "context": SAMPLE_MEDICAL_RECORD,
                    "template_id": "shinkinen",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["template_id"] == "shinkinen"

    def test_ask_endpoint_default_template(self, mock_llm):
        """template_id 省略時にデフォルトテンプレートが使われること。"""
        from app import app

        with TestClient(app) as client:
            response = client.post(
                "/ask",
                json={"context": SAMPLE_MEDICAL_RECORD},
            )

        assert response.status_code == 200
        data = response.json()
        # HOSPITAL 環境変数のデフォルトは "hanwa"
        assert data["template_id"] == "hanwa"
