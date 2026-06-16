"""検索ノード。

LLM が検索ツールを自律的に選択・実行する Agentic Search の核心。
ベクトル DB は使わず、キーワード検索・日付範囲検索のツールを提供する。
gpt-oss:120b の tool calling が不安定な場合のフォールバックも実装する。
"""

import json

from graph.state import NursingSummaryState
from llm.client import chat_with_tools
from llm.prompts import SEARCH_PROMPT

# --- 検索ツール定義 ---


def search_by_keyword(keyword: str) -> list[str]:
    """キーワードで医療記録チャンクを検索する。

    Args:
        keyword: 検索キーワード（例: "バイタル", "SpO2"）。

    Returns:
        キーワードを含むチャンクのリスト。
    """
    # 実際の実行は _execute_keyword_search で行う（チャンクは state から注入）
    return []


def search_by_date_range(start_date: str, end_date: str) -> list[str]:
    """日付範囲で医療記録チャンクを検索する。

    Args:
        start_date: 開始日（YYYYMMDD 形式）。
        end_date: 終了日（YYYYMMDD 形式）。

    Returns:
        指定期間のチャンクリスト。
    """
    return []


def _execute_keyword_search(
    keyword: str,
    chunks: list[str],
) -> list[str]:
    """キーワード検索の実行。"""
    keyword_lower = keyword.lower()
    return [c for c in chunks if keyword_lower in c.lower()]


def _execute_date_range_search(
    start_date: str,
    end_date: str,
    chunks: list[str],
    chunk_index: list[dict],
) -> list[str]:
    """日付範囲検索の実行。"""
    results = []
    for i, meta in enumerate(chunk_index):
        date = meta.get("date", "")
        if date and start_date <= date <= end_date:
            if i < len(chunks):
                results.append(chunks[i])
    return results


def _execute_tool_call(
    func_name: str,
    arguments: dict,
    chunks: list[str],
    chunk_index: list[dict],
) -> list[str]:
    """ツール呼び出しを実行する。"""
    if func_name == "search_by_keyword":
        keyword = arguments.get("keyword", "")
        return _execute_keyword_search(keyword, chunks)
    elif func_name == "search_by_date_range":
        start = arguments.get("start_date", "")
        end = arguments.get("end_date", "")
        return _execute_date_range_search(start, end, chunks, chunk_index)
    return []


def search(state: NursingSummaryState) -> dict:
    """LLM が検索ツールを自律的に選択・実行する。

    まず Ollama の tool calling を試み、失敗した場合は
    検索計画のクエリでキーワード検索にフォールバックする。
    """
    section = state["search_plan"][state["current_section_idx"]]
    chunks = state["chunks"]
    chunk_index = state["chunk_index"]

    # 日付範囲の取得
    dates = sorted(
        set(c.get("date", "") for c in chunk_index if c.get("date"))
    )
    date_range = f"{dates[0]}〜{dates[-1]}" if dates else "不明"

    results = []

    # tool calling を試みる
    try:
        prompt = SEARCH_PROMPT.format(
            section_name=section["section_name"],
            search_description=section.get("description", ""),
            date_range=date_range,
            chunk_count=len(chunks),
        )
        response = chat_with_tools(
            prompt,
            tools=[search_by_keyword, search_by_date_range],
        )

        if response.message.tool_calls:
            for tool_call in response.message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = tool_call.function.arguments
                if isinstance(fn_args, str):
                    fn_args = json.loads(fn_args)
                tool_results = _execute_tool_call(
                    fn_name,
                    fn_args,
                    chunks,
                    chunk_index,
                )
                results.extend(tool_results)
    except Exception:
        # tool calling が失敗した場合はフォールバック
        pass

    # フォールバック: search_plan のクエリでキーワード検索
    if not results:
        queries = section.get("search_queries", [])
        for query in queries:
            found = _execute_keyword_search(query, chunks)
            results.extend(found)

    # 重複除去（順序保持）
    seen = set()
    unique_results = []
    for r in results:
        r_hash = hash(r)
        if r_hash not in seen:
            seen.add(r_hash)
            unique_results.append(r)

    # 結果数の上限（LLM コンテキスト超過防止）
    max_results = 5
    unique_results = unique_results[:max_results]

    return {"_search_results": unique_results}
