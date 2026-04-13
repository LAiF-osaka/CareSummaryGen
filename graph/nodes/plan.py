"""検索計画ノード。

テンプレートのセクション定義から、医療記録を検索するための
具体的なクエリを LLM で生成する。
"""

import json

from graph.state import NursingSummaryState
from llm.client import chat
from llm.prompts import PLAN_PROMPT


def plan(state: NursingSummaryState) -> dict:
    """現セクションに必要な情報の検索クエリを LLM で生成する。

    テンプレートのセクション定義を元に、医療記録から
    どのようなキーワードで検索すべきかの計画を立てる。
    """
    section = state["search_plan"][state["current_section_idx"]]

    # 利用可能な日付一覧を取得
    dates = sorted(set(
        c.get("date", "unknown") for c in state["chunk_index"]
    ))
    available_dates = ", ".join(dates) if dates else "不明"

    prompt = PLAN_PROMPT.format(
        section_name=section["section_name"],
        section_description=section.get("description", ""),
        available_dates=available_dates,
    )

    # 構造化出力でクエリリストを取得
    schema = {
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "検索キーワードのリスト",
            }
        },
        "required": ["queries"],
    }

    try:
        response_text = chat(prompt, format_schema=schema, temperature=0.0)
        result = json.loads(response_text)
        queries = result.get("queries", [])
    except (json.JSONDecodeError, KeyError):
        # 構造化出力が失敗した場合はテキストからキーワードを抽出
        response_text = chat(prompt, temperature=0.0)
        queries = _extract_keywords_from_text(response_text)

    # search_plan を更新
    updated_plan = [s.copy() for s in state["search_plan"]]
    updated_plan[state["current_section_idx"]]["search_queries"] = queries

    return {
        "search_plan": updated_plan,
        "search_iteration": 0,
    }


def _extract_keywords_from_text(text: str) -> list[str]:
    """テキストからキーワードを抽出するフォールバック。

    箇条書きの行をキーワードとして扱う。
    """
    keywords = []
    for line in text.split("\n"):
        line = line.strip().lstrip("- •*").strip()
        if line and len(line) < 50:
            keywords.append(line)
    return keywords[:5]
