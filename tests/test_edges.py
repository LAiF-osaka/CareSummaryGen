"""条件付きエッジ関数のテスト。"""

from graph.edges.reflection import should_continue_reflection
from graph.edges.search_loop import evaluate_sufficiency, has_more_sections


def _base_state(**overrides):
    """テスト用の基本 state を構築する。"""
    state = {
        "search_iteration": 0,
        "max_search_iterations": 3,
        "_section_sufficient": False,
        "current_section_idx": 0,
        "search_plan": [{"section_key": "a"}, {"section_key": "b"}],
        "reflection_approved": False,
        "iteration_count": 0,
        "max_iterations": 2,
    }
    state.update(overrides)
    return state


# --- evaluate_sufficiency ---

def test_evaluate_sufficient():
    """情報が十分なら next_section を返すこと。"""
    state = _base_state(_section_sufficient=True)
    assert evaluate_sufficiency(state) == "next_section"


def test_evaluate_max_iterations():
    """検索上限到達で next_section を返すこと。"""
    state = _base_state(search_iteration=3, max_search_iterations=3)
    assert evaluate_sufficiency(state) == "next_section"


def test_evaluate_continue_search():
    """不足 & 上限未到達で search を返すこと。"""
    state = _base_state(search_iteration=1)
    assert evaluate_sufficiency(state) == "search"


# --- has_more_sections ---

def test_has_more_sections_yes():
    """未処理セクションがあれば plan を返すこと。"""
    state = _base_state(current_section_idx=0)
    assert has_more_sections(state) == "plan"


def test_has_more_sections_done():
    """全セクション完了で synthesize を返すこと。"""
    state = _base_state(current_section_idx=2)
    assert has_more_sections(state) == "synthesize"


# --- should_continue_reflection ---

def test_reflection_approved():
    """APPROVED なら output_formatter を返すこと。"""
    state = _base_state(reflection_approved=True, iteration_count=1)
    assert should_continue_reflection(state) == "output_formatter"


def test_reflection_max_reached():
    """上限到達で output_formatter を返すこと。"""
    state = _base_state(iteration_count=2, max_iterations=2)
    assert should_continue_reflection(state) == "output_formatter"


def test_reflection_continue():
    """問題あり & 上限未到達で revise を返すこと。"""
    state = _base_state(iteration_count=1, max_iterations=2)
    assert should_continue_reflection(state) == "revise"
