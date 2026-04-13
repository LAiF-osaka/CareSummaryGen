"""input_adapter ノードのテスト。"""

from graph.nodes.input_adapter import input_adapter, _build_search_index


def test_build_search_index_with_dates():
    """日付パターンでチャンク分割されること。"""
    text = """# 患者ID: 12345

- 20230209
  - カルテ#1
    入院時バイタル BP 120/80

- 20230210
  - カルテ#1
    状態安定

- 20230211
  - カルテ#1
    退院準備
"""
    chunks, index = _build_search_index(text)
    assert len(chunks) == 3
    assert index[0]["date"] == "20230209"
    assert index[1]["date"] == "20230210"
    assert index[2]["date"] == "20230211"
    assert "BP 120/80" in chunks[0]


def test_build_search_index_no_dates():
    """日付パターンがない場合にトークンベースでフォールバックすること。"""
    text = "これは日付パターンのないテキストです。" * 10
    chunks, index = _build_search_index(text)
    assert len(chunks) >= 1
    assert index[0]["date"] == "unknown"


def test_input_adapter_creates_search_plan():
    """input_adapter がテンプレートからセクション計画を生成すること。"""
    state = {
        "patient_id": "test",
        "raw_context": "- 20230209\n  テスト記録",
        "hospital": "hanwa",
        "template_id": "hanwa",
        "template": {},
        "chunks": [],
        "chunk_index": [],
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
    result = input_adapter(state)
    assert len(result["search_plan"]) == 6  # 阪和病院は6セクション
    assert result["search_plan"][0]["section_key"] == "instruction"
    assert len(result["chunks"]) >= 1
