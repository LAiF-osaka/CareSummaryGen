"""LangGraph State 定義（v2: agentic search 再設計）。

入力規模に応じて single-pass（≤閾値で全セクション1回生成）と
section-routed map（>閾値でセクション単位）を切り替えるグラフの状態。
詳細設計: docs/agentic-search-redesign.md を参照。
"""

from operator import add
from typing import Annotated, Optional, TypedDict


def merge_sections(left: dict, right: dict) -> dict:
    """section_key 単位で結果を集約する reducer。

    各 section_worker は自分の section_key だけを書くため衝突しない。
    single-pass の結果を section_worker の再生成で上書きする用途にも使う。
    """
    return {**left, **right}


class SectionResult(TypedDict, total=False):
    """1セクションの生成結果。

    Attributes:
        section_key: テンプレートのセクションキー。
        body: セクション本文。
        cited_dates: 本文が根拠とした日付（YYYYMMDD）。
        missing: 未充足のカテゴリ・項目（finalize で明示・review 対象）。
        review_flag: 人手レビューが必要か。
        search_trace: 補完検索（agentic ②）の監査トレース。各ステップの観測・
            クエリ・追加件数・停止理由を記録する（observability）。
            single_pass や補完不要セクションでは空。
    """

    section_key: str
    body: str
    cited_dates: list[str]
    missing: list[str]
    review_flag: bool
    search_trace: list[dict]


class GlobalState(TypedDict):
    """看護サマリー生成グラフ（v2）のメイン状態。"""

    # --- 入力 ---
    patient_id: str
    raw_context: str
    hospital: str
    template_id: str

    # --- ingest 成果物 ---
    template: dict
    routing: dict
    summary_header: str
    chunks: list[str]
    grep_index: list[dict]
    total_tokens: int

    # --- セクション結果（並列集約） ---
    section_results: Annotated[dict[str, SectionResult], merge_sections]

    # --- 出力 ---
    draft_summary: str
    final_summary: str
    review_flags: Annotated[list[str], add]
    error: Optional[str]
