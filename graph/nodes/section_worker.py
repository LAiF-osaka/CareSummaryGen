"""section_worker ノード（v2・ハイブリッド検索）。

Send で起動される単一ノード。1セクション分を次の順で処理し、親
GlobalState.section_results に {section_key: SectionResult} を返す
（reducer がマージ）。

  1. collect()            … 決定論的全件収集（安全網・LLM不使用）
  2. supplemental_search  … LLM が不足を判断し追加クエリを動的生成（agentic 部分）
  3. _extract()           … 本文生成（空なら最大 MAX_REFILL 回再試行）
  4. absent_categories()  … 決定論的網羅点検（LLMの停止判断に依存しない）

LLM のクエリ生成・停止判断は agentic だが、grep 実行と網羅点検は決定論で、
反復上限・新規スパンゼロ検出・収集の先行・点検の後行という決定論ガードレールで
囲む（verifier-in-the-loop）。詳細: docs/agentic-search-redesign.md。
"""

from config.settings import LARGE_NUM_CTX, MAX_REFILL, MAX_SEARCH_STEPS
from graph.search_index import (
    absent_categories,
    collect,
    execute_search_tool,
)
from graph.state import SectionResult
from llm.client import chat_json
from llm.prompts import SECTION_EXTRACT_PROMPT, SUPPLEMENT_PROMPT

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "body": {"type": "string"},
        "cited_dates": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["body"],
}

_SUPPLEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "need_more": {"type": "boolean"},
        "tool": {"type": "string"},
        "keyword": {"type": "string"},
        "start_date": {"type": "string"},
        "end_date": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["need_more"],
}


def section_worker(payload: dict) -> dict:
    """1セクションをハイブリッド検索で生成する（Send で起動）。

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

    # 1. 決定論的全件収集（安全網）
    collected, present_labels, _dates = collect(entry, grep_index, chunks)

    # 2. LLM 補完検索ループ（agentic 部分・決定論ガードレール内）
    collected = _supplemental_search(section, entry, collected, grep_index)

    # synthetic は全チャンク供給で num_ctx を拡張する
    override = (
        {"num_ctx": LARGE_NUM_CTX}
        if entry.get("mode") == "synthetic"
        else None
    )

    # 3. 生成（本文が空なら決定論カウンタで再試行）
    body, cited = "", []
    for _attempt in range(MAX_REFILL + 1):
        body, cited = _extract(section, collected, summary_header, override)
        if body.strip():
            break

    # 4. 決定論的網羅点検（LLM の停止判断に依存しない）
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


def _supplemental_search(
    section: dict,
    entry: dict,
    collected: list[str],
    grep_index: list[dict],
) -> list[str]:
    """LLM が不足を判断し追加検索クエリを動的生成する補完ループ。

    クエリ生成と停止判断は LLM（agentic）、grep 実行は決定論。
    決定論ガードレール: 反復上限 MAX_SEARCH_STEPS、新規スパンゼロで停止、
    chat_json 失敗（None）でループを抜けて決定論モードへフォールバック。

    synthetic（全チャンク供給）は補完不要のためスキップする。

    Args:
        section: テンプレートのセクション定義。
        entry: routing のセクションエントリ。
        collected: 決定論収集済みの evidence テキスト。
        grep_index: スパン索引。

    Returns:
        補完を加えた evidence テキストのリスト。
    """
    if entry.get("mode") == "synthetic":
        return collected

    available_categories = sorted(
        {s["category_label"] for s in grep_index if s["category_label"]}
    )
    available_dates = sorted({s["date"] for s in grep_index if s.get("date")})
    seen = set(collected)

    for _step in range(MAX_SEARCH_STEPS):
        prompt = SUPPLEMENT_PROMPT.format(
            section_name=section["name"],
            section_description=section.get("description", ""),
            collected_summary=_summarize_collected(collected),
            available_categories="、".join(available_categories) or "なし",
            available_dates="、".join(available_dates) or "なし",
        )
        decision = chat_json(prompt, _SUPPLEMENT_SCHEMA)
        if not decision or not decision.get("need_more"):
            break  # LLM の停止判断、または構造化出力失敗（フォールバック）

        tool = str(decision.get("tool", "")).strip()
        new_spans = execute_search_tool(tool, decision, grep_index)
        added = [s for s in new_spans if s not in seen]
        if not added:
            break  # 限界効用ゼロ（決定論ガードレール）
        collected = collected + added
        seen.update(added)

    return collected


def _summarize_collected(collected: list[str]) -> str:
    """収集済み evidence を補完判断用に要約する（先頭行の列挙）。"""
    if not collected:
        return "（まだ何も収集していない）"
    heads = []
    for text in collected[:12]:
        first = text.strip().split("\n", 1)[0]
        heads.append(f"- {first}")
    return "\n".join(heads)


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
    parsed = chat_json(prompt, _EXTRACT_SCHEMA, options_override=override)
    if not parsed:
        return "", []
    body = str(parsed.get("body", "")).strip()
    cited = [str(d) for d in parsed.get("cited_dates", []) if d]
    return body, cited
