"""LangGraph グラフ構築モジュール。

Agentic Search + Reflection のグラフを構築・コンパイルする。
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from graph.edges.reflection import should_continue_reflection
from graph.edges.search_loop import (
    evaluate_sufficiency,
    has_more_sections,
    next_section,
)
from graph.nodes.evaluate import evaluate
from graph.nodes.extract import extract
from graph.nodes.input_adapter import input_adapter
from graph.nodes.output_formatter import output_formatter
from graph.nodes.plan import plan
from graph.nodes.reflect import reflect
from graph.nodes.revise import revise
from graph.nodes.search import search
from graph.nodes.synthesize import synthesize
from graph.state import NursingSummaryState


def build_nursing_summary_graph() -> CompiledStateGraph:
    """看護サマリー生成グラフを構築してコンパイルする。

    Returns:
        コンパイル済みの LangGraph グラフ。
    """
    builder = StateGraph(NursingSummaryState)

    # --- ノード登録 ---
    builder.add_node("input_adapter", input_adapter)
    builder.add_node(
        "plan", plan, retry_policy=RetryPolicy(max_attempts=2)
    )
    builder.add_node("search", search)
    builder.add_node(
        "extract", extract, retry_policy=RetryPolicy(max_attempts=2)
    )
    builder.add_node("evaluate", evaluate)
    builder.add_node("next_section", next_section)
    builder.add_node(
        "synthesize", synthesize, retry_policy=RetryPolicy(max_attempts=2)
    )
    builder.add_node("reflect", reflect)
    builder.add_node("revise", revise)
    builder.add_node("output_formatter", output_formatter)

    # --- エッジ定義 ---

    # 入力 → 計画
    builder.add_edge(START, "input_adapter")
    builder.add_edge("input_adapter", "plan")

    # Agentic Search ループ
    builder.add_edge("plan", "search")
    builder.add_edge("search", "extract")
    builder.add_edge("extract", "evaluate")

    # evaluate → 再検索 or 次セクション
    builder.add_conditional_edges(
        "evaluate",
        evaluate_sufficiency,
        {"search": "search", "next_section": "next_section"},
    )

    # next_section → 次セクション or 統合
    builder.add_conditional_edges(
        "next_section",
        has_more_sections,
        {"plan": "plan", "synthesize": "synthesize"},
    )

    # 統合 → Reflection
    builder.add_edge("synthesize", "reflect")

    # Reflection ループ
    builder.add_conditional_edges(
        "reflect",
        should_continue_reflection,
        {"revise": "revise", "output_formatter": "output_formatter"},
    )
    builder.add_edge("revise", "reflect")

    # 出力
    builder.add_edge("output_formatter", END)

    return builder.compile()
