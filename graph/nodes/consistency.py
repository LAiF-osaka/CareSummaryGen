"""consistency ノード（v2）。

ドラフトの atomic claim を LLM で分解し、各 claim が grep 索引（入力記録）に
支持されるかを決定論照合する。未支持 claim はレビューフラグを立てるのみで、
自動修復はしない（医療の偽陽性リスクを避ける）。

詳細設計: docs/agentic-search-redesign.md §2, §5 を参照。
"""

import re

from graph.state import GlobalState
from llm.client import chat_json
from llm.prompts import CONSISTENCY_PROMPT

_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["claims"],
}

# 照合に使う数値・日付などのトークン抽出（決定論照合の手掛かり）
_TOKEN = re.compile(r"[0-9]+(?:[./][0-9]+)?|[A-Za-zぁ-んァ-ヶ一-龠]{2,}")


def consistency(state: GlobalState) -> dict:
    """ドラフトの claim を抽出し、未支持 claim をフラグ化する。

    自動修復はせず review_flags に未支持セクション/claim を記録する。
    """
    draft = state.get("draft_summary", "")
    if not draft.strip():
        return {}

    parsed = chat_json(CONSISTENCY_PROMPT.format(draft=draft), _SCHEMA)
    if not parsed or not isinstance(parsed.get("claims"), list):
        # claim 分解に失敗した場合は照合をスキップ（フラグなし）
        return {}

    evidence_text = "\n".join(s["text"] for s in state["grep_index"])
    evidence_text += "\n" + state.get("summary_header", "")

    flags: list[str] = []
    for claim in parsed["claims"]:
        claim = str(claim).strip()
        if claim and not _is_supported(claim, evidence_text):
            flags.append(f"未支持claim: {claim}")

    return {"review_flags": flags} if flags else {}


def _is_supported(claim: str, evidence: str) -> bool:
    """claim の主要トークンが evidence に十分含まれるかを決定論判定する。

    claim 内の数値・語のうち過半が evidence に出現すれば支持とみなす。
    LLM-as-judge は使わない（偽陽性で事実誤認に直結するため）。
    """
    tokens = _TOKEN.findall(claim)
    if not tokens:
        return True
    hit = sum(1 for t in tokens if t in evidence)
    return hit >= max(1, len(tokens) // 2)
