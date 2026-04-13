"""Reflection ノード。

ドラフトサマリーの具体的な問題点をテキストで指摘する。
LLM スコアやルールベース検証は使用しない。
問題がなければ "APPROVED" を返し早期終了のトリガーとなる。
"""

from graph.state import NursingSummaryState
from llm.client import chat
from llm.prompts import REFLECTION_PROMPT


def reflect(state: NursingSummaryState) -> dict:
    """ドラフトサマリーの問題点をテキストで指摘する。

    問題がなければ LLM は "APPROVED" を返す。
    LLM 呼び出しが失敗した場合はそのまま承認にフォールバック。
    """
    sections = state["template"].get("sections", [])
    section_names = "\n".join(f"- {s['name']}" for s in sections)

    prompt = REFLECTION_PROMPT.format(
        template_sections=section_names,
        draft_summary=state["draft_summary"],
    )

    try:
        feedback = chat(prompt, temperature=0.0)
        approved = "APPROVED" in feedback.upper()
    except Exception:
        # LLM 呼び出し失敗時はそのまま承認
        feedback = ""
        approved = True

    return {
        "reflection_feedback": feedback,
        "reflection_approved": approved,
        "iteration_count": state["iteration_count"] + 1,
    }
