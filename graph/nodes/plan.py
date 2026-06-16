"""検索計画ノード。

テンプレートのセクション定義から、医療記録を検索するための
具体的なクエリを LLM で生成する。
"""

from graph.state import NursingSummaryState
from llm.client import chat, extract_json
from llm.prompts import PLAN_PROMPT


def plan(state: NursingSummaryState) -> dict:
    """現セクションに必要な情報の検索クエリを LLM で生成する。

    テンプレートのセクション定義を元に、医療記録から
    どのようなキーワードで検索すべきかの計画を立てる。
    """
    section = state["search_plan"][state["current_section_idx"]]

    # 利用可能な日付一覧を取得
    dates = sorted(set(c.get("date", "unknown") for c in state["chunk_index"]))
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

    response_text = chat(prompt, format_schema=schema, temperature=0.0)

    # gpt-oss は format 指定でも reasoning trace 等を伴うことがあるため
    # 堅牢な JSON 抽出を行い、失敗時はテキストからキーワードを拾う。
    result = extract_json(response_text)
    if result and isinstance(result.get("queries"), list):
        queries = [str(q).strip() for q in result["queries"] if str(q).strip()]
    else:
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

    JSON 抽出に失敗した場合に使用する。Markdown の表・見出し・装飾行を
    除外し、短い単語・語句のみをキーワードとして拾う。

    Args:
        text: LLM の応答テキスト。

    Returns:
        キーワード候補（最大5件）。
    """
    keywords: list[str] = []
    for raw in text.split("\n"):
        # 箇条書き記号・番号・装飾を除去
        line = raw.strip().lstrip("-•*0123456789.（）()# ").strip()
        line = line.strip("*`「」 　")
        if not line:
            continue
        # Markdown 表・見出し・コロン終端の見出しを除外
        if "|" in raw or raw.lstrip().startswith(("#", "|", "**")):
            continue
        if line.endswith(":") or line.endswith("："):
            continue
        # 長すぎる行（文章）は検索キーワードとして不適
        if len(line) >= 20:
            continue
        keywords.append(line)
    return keywords[:5]
