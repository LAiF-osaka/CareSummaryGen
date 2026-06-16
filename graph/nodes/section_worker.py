"""section_worker ノード（v2）。

Send で起動される単一ノード。1セクション分の collect → extract → verify →
refill（最大1）を関数内で完結し、親 GlobalState.section_results に
{section_key: SectionResult} を返す（reducer がマージ）。

サブグラフではなく単一ノードにすることで、Send 経由でも結果が確実に
親 state へ集約される。停止判断に LLM スコアは使わない（決定論）。

詳細設計: docs/agentic-search-redesign.md §2, §4, §8 を参照。
"""

from config.settings import LARGE_NUM_CTX, MAX_REFILL
from graph.search_index import absent_categories, collect
from graph.state import SectionResult
from llm.client import chat_json
from llm.prompts import SECTION_EXTRACT_PROMPT

_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "body": {"type": "string"},
        "cited_dates": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["body"],
}


def section_worker(payload: dict) -> dict:
    """1セクションを生成する（Send で起動される単一ノード）。

    Args:
        payload: {section, routing_entry, grep_index, chunks, summary_header}。

    Returns:
        {"section_results": {section_key: SectionResult}}。
    """
    section = payload["section"]
    entry = payload["routing_entry"]
    grep_index = payload["grep_index"]
    chunks = payload["chunks"]
    summary_header = payload["summary_header"]

    collected, present_labels, _dates = collect(entry, grep_index, chunks)
    # synthetic は全チャンク供給で num_ctx を拡張する
    override = (
        {"num_ctx": LARGE_NUM_CTX}
        if entry.get("mode") == "synthetic"
        else None
    )

    body, cited = "", []
    for _attempt in range(MAX_REFILL + 1):
        body, cited = _extract(section, collected, summary_header, override)
        if body.strip():
            break
        # body が空（抽出失敗）の場合のみ再試行（決定論カウンタ）

    absent = absent_categories(entry, present_labels)
    review = bool(absent) or not body.strip()

    key = section["key"]
    return {
        "section_results": {
            key: SectionResult(
                section_key=key,
                body=body.strip() or "記録なし（要確認）",
                cited_dates=cited,
                missing=absent,
                review_flag=review,
            )
        }
    }


def _extract(
    section: dict,
    collected: list[str],
    summary_header: str,
    override: dict | None,
) -> tuple[str, list[str]]:
    """収集 evidence からセクション本文を生成する。"""
    evidence = "\n\n---\n\n".join(collected) if collected else "記録なし"
    prompt = SECTION_EXTRACT_PROMPT.format(
        section_name=section["name"],
        section_description=section.get("description", ""),
        summary_header=summary_header or "記録なし",
        evidence=evidence,
    )
    parsed = chat_json(prompt, _SCHEMA, options_override=override)
    if not parsed:
        return "", []
    body = str(parsed.get("body", "")).strip()
    cited = [str(d) for d in parsed.get("cited_dates", []) if d]
    return body, cited
