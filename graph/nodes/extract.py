"""情報抽出ノード。

検索結果からセクションに必要な情報を LLM で抽出する。
"""

from graph.state import NursingSummaryState
from llm.client import chat
from llm.prompts import EXTRACT_PROMPT


def extract(state: NursingSummaryState) -> dict:
    """検索結果からセクションに必要な情報を LLM で抽出する。

    search ノードが返したチャンクを LLM に渡し、
    セクション定義に基づいて必要な情報を抽出する。
    検索結果が空の場合は「記録なし」を返す。
    """
    section = state["search_plan"][state["current_section_idx"]]
    search_results = state.get("_search_results", [])

    if not search_results:
        # 検索結果なし
        section_key = section["section_key"]
        return {
            "section_results": {section_key: "記録なし"},
            "search_iteration": state["search_iteration"] + 1,
        }

    # 検索結果を結合してプロンプトに渡す
    combined_results = "\n\n---\n\n".join(search_results)

    prompt = EXTRACT_PROMPT.format(
        section_name=section["section_name"],
        section_description=section.get("description", ""),
        search_results=combined_results,
    )

    response_text = chat(prompt)

    section_key = section["section_key"]
    return {
        "section_results": {section_key: response_text},
        "search_iteration": state["search_iteration"] + 1,
    }
