"""LangGraph グラフ構築（v2 agentic search）。

入力規模で single-pass / section-routed map を切り替え、
セクションを Send で並列処理し、決定論的に組み立てる。

詳細設計: docs/agentic-search-redesign.md を参照。

フロー:
    ingest → [route_and_fanout]
        ≤閾値 → single_pass → [after_single_pass]
                → (欠損のみ)section_worker → assemble
        >閾値 → section_worker(Send×N) → assemble
    assemble → consistency → finalize → END
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import RetryPolicy

from graph.edges.routing_v2 import after_single_pass, route_and_fanout
from graph.nodes.assemble import assemble
from graph.nodes.consistency import consistency
from graph.nodes.finalize import finalize
from graph.nodes.ingest import ingest
from graph.nodes.section_worker import section_worker
from graph.nodes.single_pass import single_pass
from graph.state import GlobalState


def build_nursing_summary_graph() -> CompiledStateGraph:
    """看護サマリー生成グラフ（v2）を構築してコンパイルする。"""
    builder = StateGraph(GlobalState)

    _llm_retry = RetryPolicy(max_attempts=2)

    builder.add_node("ingest", ingest)
    builder.add_node("single_pass", single_pass, retry_policy=_llm_retry)
    builder.add_node("section_worker", section_worker, retry_policy=_llm_retry)
    builder.add_node("assemble", assemble)
    builder.add_node("consistency", consistency)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "ingest")

    # 入力規模で分岐: single_pass（≤閾値）or section_worker への Send（>閾値）
    builder.add_conditional_edges(
        "ingest",
        route_and_fanout,
        ["single_pass", "section_worker"],
    )

    # single_pass 後: 欠損セクションのみ section_worker、無ければ assemble
    builder.add_conditional_edges(
        "single_pass",
        after_single_pass,
        ["section_worker", "assemble"],
    )

    # section_worker 完了（reducer 集約）→ assemble
    builder.add_edge("section_worker", "assemble")

    builder.add_edge("assemble", "consistency")
    builder.add_edge("consistency", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()
