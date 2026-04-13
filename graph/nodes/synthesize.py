"""統合生成ノード。

全セクションの抽出結果をテンプレートに沿って統合し、
看護サマリーのドラフトを生成する。
"""

from graph.state import NursingSummaryState
from llm.client import chat
from llm.prompts import SYNTHESIZE_PROMPT
from templates_loader.loader import build_format_instruction


def synthesize(state: NursingSummaryState) -> dict:
    """全セクションの抽出結果をテンプレートに沿って統合ドラフトを生成する。"""
    format_instruction = build_format_instruction(state["template"])

    # セクション別情報をフォーマット
    section_data_parts = []
    for section in state["template"]["sections"]:
        key = section["key"]
        content = state["section_results"].get(key, "記録なし")
        section_data_parts.append(f"### {section['name']}\n{content}")

    section_data = "\n\n".join(section_data_parts)

    prompt = SYNTHESIZE_PROMPT.format(
        format_instruction=format_instruction,
        section_data=section_data,
    )

    response_text = chat(prompt)

    return {"draft_summary": response_text}
