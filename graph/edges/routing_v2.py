"""v2 グラフの条件分岐・ファンアウト。

入力規模による経路選択（single_pass / fanout）と、欠損セクションの
section_worker への Send を定義する。

詳細設計: docs/agentic-search-redesign.md §2 を参照。
"""

from langgraph.types import Send

from config.settings import SINGLE_PASS_TOKEN_THRESHOLD
from graph.state import GlobalState


def route_and_fanout(state: GlobalState) -> str | list[Send]:
    """総トークンが閾値以下なら single_pass、超なら全セクションを Send。

    Returns:
        "single_pass"（≤閾値）または section_worker への Send リスト（>閾値）。
    """
    if state["total_tokens"] <= SINGLE_PASS_TOKEN_THRESHOLD:
        return "single_pass"
    return [_section_send(state, s) for s in state["template"]["sections"]]


def _section_send(state: GlobalState, section: dict) -> Send:
    """1セクションを section_worker へ Send するペイロードを構築する。"""
    return Send(
        "section_worker",
        {
            "section": section,
            "routing_entry": state["routing"].get(section["key"], {}),
            "grep_index": state["grep_index"],
            "chunks": state["chunks"],
            "summary_header": state["summary_header"],
        },
    )


def after_single_pass(state: GlobalState) -> str | list[Send]:
    """single_pass の結果から欠損セクションのみ section_worker へ回す。

    欠損が無ければ "assemble" を返す（直接組立へ）。
    """
    deficient = [
        s
        for s in state["template"]["sections"]
        if not _has_body(state, s["key"])
    ]
    if not deficient:
        return "assemble"
    return [_section_send(state, s) for s in deficient]


def _has_body(state: GlobalState, key: str) -> bool:
    """指定セクションに本文が生成済みかを返す。"""
    result = state["section_results"].get(key)
    return bool(result and result.get("body"))
