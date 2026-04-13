"""Reflection ループの条件付きエッジ関数。

LLM スコアやルールベース判定は使用しない。
LLM が APPROVED と判断したら早期終了、問題ありなら上限まで revise を続行する。
"""

from typing import Literal

from graph.state import NursingSummaryState


def should_continue_reflection(
    state: NursingSummaryState,
) -> Literal["revise", "output_formatter"]:
    """Reflection ループの継続を判定する。

    - LLM が APPROVED → 早期終了
    - 上限到達 → 終了
    - 問題あり & 上限未到達 → revise 継続
    """
    if state["reflection_approved"]:
        return "output_formatter"

    if state["iteration_count"] >= state["max_iterations"]:
        return "output_formatter"

    return "revise"
