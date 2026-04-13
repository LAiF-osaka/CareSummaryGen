"""出力フォーマッターノード。

Reflection を通過したドラフトを最終サマリーとして確定する。
"""

from graph.state import NursingSummaryState


def output_formatter(state: NursingSummaryState) -> dict:
    """ドラフトを最終サマリーとして確定する。"""
    return {"final_summary": state["draft_summary"]}
