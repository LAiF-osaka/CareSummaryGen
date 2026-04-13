"""State 定義のテスト。"""

from graph.state import NursingSummaryState, merge_dicts


def test_merge_dicts_new_key():
    """新しいキーが追加されること。"""
    left = {"a": "hello"}
    right = {"b": "world"}
    result = merge_dicts(left, right)
    assert result == {"a": "hello", "b": "world"}


def test_merge_dicts_append():
    """既存キーの値が追記されること。"""
    left = {"a": "first"}
    right = {"a": "second"}
    result = merge_dicts(left, right)
    assert result["a"] == "first\nsecond"


def test_merge_dicts_empty_left():
    """左が空でも正常動作すること。"""
    result = merge_dicts({}, {"a": "value"})
    assert result == {"a": "value"}


def test_state_has_expected_fields():
    """State に全フィールドが定義されていること。"""
    fields = list(NursingSummaryState.__annotations__.keys())
    assert "patient_id" in fields
    assert "raw_context" in fields
    assert "search_plan" in fields
    assert "section_results" in fields
    assert "reflection_approved" in fields
    assert "final_summary" in fields
