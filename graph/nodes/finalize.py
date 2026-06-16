"""finalize ノード（v2）。

ドラフトを最終サマリーとして確定し、review_flag が立ったセクションを
レビュー対象として集約する。

詳細設計: docs/agentic-search-redesign.md §2 を参照。
"""

from graph.state import GlobalState


def finalize(state: GlobalState) -> dict:
    """final_summary を確定し、レビュー対象セクションを集約する。"""
    section_flags = [
        result["section_key"]
        for result in state["section_results"].values()
        if result.get("review_flag")
    ]
    return {
        "final_summary": state.get("draft_summary", ""),
        "review_flags": section_flags,
    }
