"""single_pass ノード（v2）。

総トークンが閾値以下の場合に、全セクションを1回の LLM 呼び出しで生成する
（短〜中期入院の最安経路）。欠損・空セクションは後続の section_worker で
個別再生成される。

詳細設計: docs/agentic-search-redesign.md §1, §2 を参照。
"""

from config.settings import LARGE_NUM_CTX
from graph.state import GlobalState, SectionResult
from llm.client import chat_json
from llm.prompts import SINGLE_PASS_PROMPT

_SCHEMA = {
    "type": "object",
    "properties": {
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section_key": {"type": "string"},
                    "body": {"type": "string"},
                    "cited_dates": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["section_key", "body"],
            },
        }
    },
    "required": ["sections"],
}


def single_pass(state: GlobalState) -> dict:
    """全セクションを1回の LLM 呼び出しで生成する。

    結果を section_results に格納する。空・欠落セクションは
    after_single_pass 条件分岐で section_worker へ回される。
    """
    sections = state["template"]["sections"]
    section_specs = "\n".join(
        f"- section_key={s['key']} / 名前={s['name']}"
        f" / 説明={s.get('description', '')}"
        for s in sections
    )
    records = "\n\n".join(state["chunks"])

    prompt = SINGLE_PASS_PROMPT.format(
        summary_header=state["summary_header"] or "記録なし",
        records=records or "記録なし",
        section_specs=section_specs,
    )

    parsed = chat_json(
        prompt, _SCHEMA, options_override={"num_ctx": LARGE_NUM_CTX}
    )

    # 応答を section_key -> body のマップへ
    by_key: dict[str, dict] = {}
    if parsed and isinstance(parsed.get("sections"), list):
        for item in parsed["sections"]:
            if isinstance(item, dict) and item.get("section_key"):
                by_key[item["section_key"]] = item

    results: dict[str, SectionResult] = {}
    for s in sections:
        key = s["key"]
        item = by_key.get(key, {})
        body = str(item.get("body", "")).strip()
        cited = [str(d) for d in item.get("cited_dates", []) if d]
        results[key] = SectionResult(
            section_key=key,
            body=body,
            cited_dates=cited,
            missing=[] if body else ["<single_pass未生成>"],
            review_flag=not body,
        )

    return {"section_results": results}
