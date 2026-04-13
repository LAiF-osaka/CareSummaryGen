"""情報充足度評価ノード。

抽出結果が十分かを LLM で判定し、
不足時は追加検索クエリを生成する。
"""

from graph.state import NursingSummaryState
from llm.client import chat
from llm.prompts import EVALUATE_PROMPT


def evaluate(state: NursingSummaryState) -> dict:
    """抽出結果の充足度を LLM で判定する。

    SUFFICIENT であれば次のセクションへ進み、
    不足であれば追加検索クエリを生成して再検索を促す。
    """
    section = state["search_plan"][state["current_section_idx"]]
    section_key = section["section_key"]
    extracted = state["section_results"].get(section_key, "")

    prompt = EVALUATE_PROMPT.format(
        section_name=section["section_name"],
        section_description=section.get("description", ""),
        extracted_info=extracted,
    )

    feedback = chat(prompt, temperature=0.0)

    is_sufficient = "SUFFICIENT" in feedback.upper()

    if not is_sufficient:
        # 追加検索クエリで search_plan を更新
        new_queries = _extract_additional_queries(feedback)
        updated_plan = [s.copy() for s in state["search_plan"]]
        if new_queries:
            updated_plan[state["current_section_idx"]]["search_queries"] = (
                new_queries
            )
        return {
            "search_plan": updated_plan,
            "_section_sufficient": False,
        }

    return {"_section_sufficient": True}


def _extract_additional_queries(feedback: str) -> list[str]:
    """フィードバックテキストから追加検索クエリを抽出する。"""
    queries = []
    for line in feedback.split("\n"):
        line = line.strip().lstrip("- •*").strip()
        if line and len(line) < 80 and "SUFFICIENT" not in line.upper():
            queries.append(line)
    return queries[:5]
