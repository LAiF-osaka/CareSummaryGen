"""Agentic Search ループの条件付きエッジ関数。

検索の充足度判定とセクション進行を制御する。
"""

from typing import Literal

from graph.state import NursingSummaryState


def evaluate_sufficiency(
    state: NursingSummaryState,
) -> Literal["search", "next_section"]:
    """検索を続行するか次のセクションに進むかを判定する。

    情報が十分であるか、検索上限に達した場合は次のセクションへ。
    """
    if state.get("_section_sufficient", False):
        return "next_section"

    if state["search_iteration"] >= state["max_search_iterations"]:
        return "next_section"

    return "search"


def next_section(state: NursingSummaryState) -> dict:
    """次のセクションに進む。"""
    return {
        "current_section_idx": state["current_section_idx"] + 1,
        "search_iteration": 0,
        "_search_results": [],
        "_section_sufficient": False,
    }


def has_more_sections(
    state: NursingSummaryState,
) -> Literal["plan", "synthesize"]:
    """未処理セクションが残っているかを判定する。"""
    if state["current_section_idx"] < len(state["search_plan"]):
        return "plan"
    return "synthesize"
