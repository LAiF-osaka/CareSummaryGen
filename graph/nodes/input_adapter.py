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

    # サマリヘッダ（最初の日付行より前の患者横断情報）を抽出し常時供給する。
    # 既存の日付分割では破棄される領域のため、別途 state に保持する。
    summary_header = _extract_summary_header(state["raw_context"])

    # 日付単位でチャンク分割
    chunks, chunk_index = _build_search_index(state["raw_context"])

    return {
        "template": template,
        "search_plan": search_plan,
        "summary_header": summary_header,
        "chunks": chunks,
        "chunk_index": chunk_index,
        "current_section_idx": 0,
        "search_iteration": 0,
        "section_results": {},
        "_search_results": [],
    }


def _extract_summary_header(text: str) -> str:
    """最初の `- YYYYMMDD` 日付行より前の本文をサマリヘッダとして抽出する。

    `# 患者ID:` ヘッダ行は除外し、`## サマリ基本情報` 等の患者横断情報のみを
    返す。日付行が無い、または前置領域が無い場合は空文字を返す。

    Args:
        text: 入力医療記録テキスト。

    Returns:
        サマリヘッダ本文（無ければ空文字）。
    """
    date_match = re.search(r"^- \d{8}\s*$", text, re.MULTILINE)
    preamble = text[: date_match.start()] if date_match else text

    # `# 患者ID:` 見出し行を除外し、残りを本文とする
    body_lines = [
        line
        for line in preamble.splitlines()
        if not line.lstrip().startswith("# ")
    ]
    return "\n".join(body_lines).strip()


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
