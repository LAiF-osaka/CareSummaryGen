"""LangGraph State 定義モジュール。

Agentic Search パイプラインのグラフ状態を TypedDict で定義する。
Annotated + reducer で並列ノードの結果結合戦略を制御する。
"""

from typing import Annotated, Optional, TypedDict


def merge_dicts(left: dict, right: dict) -> dict:
    """セクション結果を蓄積するカスタム reducer。

    複数ノードから同じセクションに情報が追加される場合、
    既存の情報に追記する形でマージする。
    """
    merged = {**left}
    for key, value in right.items():
        if key in merged and merged[key]:
            merged[key] = merged[key] + "\n" + value
        else:
            merged[key] = value
    return merged


class NursingSummaryState(TypedDict):
    """看護サマリー生成グラフのメイン状態。"""

    # --- 入力 ---
    patient_id: str
    raw_context: str
    hospital: str

    # --- チャンキング・検索インデックス ---
    chunks: list[str]
    chunk_index: list[dict]

    # --- テンプレート ---
    template_id: str
    template: dict

    # --- Agentic Search ---
    search_plan: list[dict]
    section_results: Annotated[dict[str, str], merge_dicts]
    current_section_idx: int
    search_iteration: int
    max_search_iterations: int

    # --- 検索中間結果（ノード間受け渡し用） ---
    _search_results: list[str]

    # --- 統合生成 ---
    draft_summary: str

    # --- Reflection ---
    reflection_feedback: str
    reflection_approved: bool
    iteration_count: int
    max_iterations: int

    # --- 出力 ---
    final_summary: str
    error: Optional[str]
