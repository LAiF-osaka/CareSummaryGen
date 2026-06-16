"""ingest ノード（v2）。

入力 Markdown を日付チャンク化し、grep 索引・サマリヘッダを構築、
テンプレートと routing をロードし、総トークン数を算出する。
LLM は使わない。

詳細設計: docs/agentic-search-redesign.md §2, §8 を参照。
"""

import re

import tiktoken

from config.settings import CHUNK_OVERLAP, CHUNK_SIZE
from graph.search_index import explode_to_spans
from graph.state import GlobalState
from templates_loader.loader import load_template
from templates_loader.routing import load_routing

_encoding = tiktoken.get_encoding("cl100k_base")

_DATE_LINE = re.compile(r"^- (\d{8})\s*$", re.MULTILINE)


def ingest(state: GlobalState) -> dict:
    """入力正規化・日付チャンク化・grep索引・routing解決・規模判定。"""
    template = load_template(state["template_id"])
    routing = load_routing(state["template_id"])

    summary_header = _extract_summary_header(state["raw_context"])
    chunks, chunk_index = _build_search_index(state["raw_context"])
    grep_index = explode_to_spans(chunks, chunk_index)
    total_tokens = len(_encoding.encode(state["raw_context"]))

    return {
        "template": template,
        "routing": routing,
        "summary_header": summary_header,
        "chunks": chunks,
        "grep_index": grep_index,
        "total_tokens": total_tokens,
        "section_results": {},
    }


def _extract_summary_header(text: str) -> str:
    """最初の `- YYYYMMDD` 日付行より前の患者横断情報を抽出する。

    `# 患者ID:` 見出し行は除外する。日付行が無ければ全体を返す。
    """
    match = _DATE_LINE.search(text)
    preamble = text[: match.start()] if match else text
    body_lines = [
        line
        for line in preamble.splitlines()
        if not line.lstrip().startswith("# ")
    ]
    return "\n".join(body_lines).strip()


def _build_search_index(text: str) -> tuple[list[str], list[dict]]:
    """医療記録を日付単位のチャンクに分割し検索インデックスを構築する。

    `- YYYYMMDD` 単独行を境界とする。日付が無い場合はトークン数で分割。
    """
    matches = list(_DATE_LINE.finditer(text))
    if matches:
        chunks: list[str] = []
        chunk_index: list[dict] = []
        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(chunk_text)
                chunk_index.append({"date": match.group(1), "index": i})
        if chunks:
            return chunks, chunk_index

    return _split_by_tokens(text)


def _split_by_tokens(text: str) -> tuple[list[str], list[dict]]:
    """トークン数ベースのフォールバック分割。"""
    tokens = _encoding.encode(text)
    if not tokens:
        return [text], [{"date": "unknown", "index": 0}]

    chunk_size = min(CHUNK_SIZE, 4096)
    overlap = min(CHUNK_OVERLAP, 200)
    step = max(1, chunk_size - overlap)

    chunks: list[str] = []
    chunk_index: list[dict] = []
    for i, start in enumerate(range(0, len(tokens), step)):
        piece = tokens[start : start + chunk_size]
        if not piece:
            break
        chunks.append(_encoding.decode(piece))
        chunk_index.append({"date": "unknown", "index": i})

    return chunks, chunk_index
