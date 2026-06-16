"""assemble ノード（v2）。

section_results をテンプレートの section_delimiter で決定論的に組み立てる
（LLM 不使用）。空・未充足セクションは「記録なし（要確認）」と明示する。

詳細設計: docs/agentic-search-redesign.md §2, §8 を参照。
"""

from graph.state import GlobalState


def assemble(state: GlobalState) -> dict:
    """section_results を決定論的に組み立ててドラフトを生成する。"""
    template = state["template"]
    delimiter = template.get("section_delimiter", "--- {name} ---")
    results = state["section_results"]

    parts: list[str] = []
    for section in template["sections"]:
        key = section["key"]
        result = results.get(key)
        body = (
            result["body"]
            if result and result.get("body")
            else "記録なし（要確認）"
        )
        parts.append(delimiter.format(name=section["name"]))
        parts.append(body)
        parts.append("")

    return {"draft_summary": "\n".join(parts).rstrip() + "\n"}
