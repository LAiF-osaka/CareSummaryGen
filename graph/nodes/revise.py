"""改善ノード。

Reflection ノードが指摘した具体的な問題箇所のみを修正する。
"""

from graph.state import NursingSummaryState
from llm.client import chat
from llm.prompts import REVISION_PROMPT


def revise(state: NursingSummaryState) -> dict:
    """フィードバックに基づいてドラフトを改善する。"""
    prompt = REVISION_PROMPT.format(
        draft_summary=state["draft_summary"],
        feedback=state["reflection_feedback"],
    )

    response_text = chat(prompt)

    return {"draft_summary": response_text}
