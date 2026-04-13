"""入力アダプターノード。

テンプレートのロード、医療記録の日付単位チャンク分割、
検索インデックスの構築を行う。
"""

import re

import tiktoken

from config.settings import CHUNK_SIZE, CHUNK_OVERLAP
from graph.state import NursingSummaryState
from templates_loader.loader import load_template

# トークナイザ（チャンク分割用の近似）
_encoding = tiktoken.get_encoding("cl100k_base")


def input_adapter(state: NursingSummaryState) -> dict:
    """テンプレートロード・チャンク分割・検索インデックス構築を行う。

    医療記録を日付単位で分割し、各チャンクにメタデータを付与する。
    Agentic Search ではチャンクは「検索対象」として使い、
    全チャンクを均一に処理するのではなく必要な箇所のみを検索する。
    """
    template = load_template(state["template_id"])

    # 検索計画の初期化（テンプレートのセクションから生成）
    search_plan = [
        {
            "section_key": section["key"],
            "section_name": section["name"],
            "description": section.get("description", ""),
            "search_queries": [],
        }
        for section in template["sections"]
    ]

    # 日付単位でチャンク分割
    chunks, chunk_index = _build_search_index(state["raw_context"])

    return {
        "template": template,
        "search_plan": search_plan,
        "chunks": chunks,
        "chunk_index": chunk_index,
        "current_section_idx": 0,
        "search_iteration": 0,
        "section_results": {},
        "_search_results": [],
    }


def _build_search_index(text: str) -> tuple[list[str], list[dict]]:
    """医療記録を日付単位のチャンクに分割し検索インデックスを構築する。

    「- YYYYMMDD」パターンを境界として分割する。
    日付パターンが見つからない場合はトークン数ベースでフォールバック。

    Returns:
        (チャンクテキストのリスト, メタデータのリスト)
    """
    # 日付パターン（"- 20230209" 等）で分割を試みる
    date_pattern = re.compile(r"^- (\d{8})\s*$", re.MULTILINE)
    matches = list(date_pattern.finditer(text))

    if matches:
        chunks = []
        chunk_index = []

        for i, match in enumerate(matches):
            start = match.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            chunk_text = text[start:end].strip()

            if chunk_text:
                date_str = match.group(1)
                chunks.append(chunk_text)
                chunk_index.append({"date": date_str, "index": i})

        if chunks:
            return chunks, chunk_index

    # フォールバック: トークン数ベースで分割
    return _split_by_tokens(text)


def _split_by_tokens(text: str) -> tuple[list[str], list[dict]]:
    """トークン数ベースのフォールバック分割。"""
    tokens = _encoding.encode(text)
    if not tokens:
        return [text], [{"date": "unknown", "index": 0}]

    chunk_size = min(CHUNK_SIZE, 4096)  # 検索用に小さめのチャンクを使用
    overlap = min(CHUNK_OVERLAP, 200)
    step = max(1, chunk_size - overlap)

    chunks = []
    chunk_index = []
    for i, start in enumerate(range(0, len(tokens), step)):
        piece = tokens[start : start + chunk_size]
        if not piece:
            break
        chunk_text = _encoding.decode(piece)
        chunks.append(chunk_text)
        chunk_index.append({"date": "unknown", "index": i})

    return chunks, chunk_index
