"""grep 索引と決定論的な収集・検証ヘルパ（v2 agentic search）。

日付チャンク Markdown を (date, category_label, text) のスパン集合へ展開し、
セクションの routing 定義に基づいてカテゴリ全件収集＋keyword grep で収集する。
ベクトル検索は使わない。LLM も使わない（決定論）。

詳細設計: docs/agentic-search-redesign.md §3, §4 を参照。
"""

import re
import unicodedata

from templates_loader.routing import resolve_category_labels

_DATE_LINE = re.compile(r"^- (\d{8})\s*$")
_CATEGORY_LINE = re.compile(r"^  - (.+?)\s*$")


def _normalize_label(label: str) -> str:
    """カテゴリラベルを照合用に正規化する（全半角・空白の揺れを吸収）。"""
    return (
        unicodedata.normalize("NFKC", label)
        .strip()
        .replace(" ", "")
        .replace("　", "")
    )


def explode_to_spans(chunks: list[str], chunk_index: list[dict]) -> list[dict]:
    """日付チャンクを (date, category_label, text) スパンへ展開する。

    DB renderer 経路では `  - {label}` カテゴリ小見出しでサブブロック化される。
    小見出しが無いチャンク（テキスト/XML 由来）は category_label=None の
    単一スパンとして扱い、keyword grep / synthetic で拾えるようにする。

    Args:
        chunks: 日付チャンク本文のリスト。
        chunk_index: 各チャンクのメタ（date 等）。

    Returns:
        スパン辞書のリスト（{date, category_label, text}）。
    """
    spans: list[dict] = []
    for chunk, meta in zip(chunks, chunk_index):
        date = meta.get("date", "unknown")
        sub_blocks = _split_category_blocks(chunk)
        has_label = any(label for label, _ in sub_blocks)
        if not has_label:
            # 小見出しなし: チャンク全体を1スパン（label=None）
            spans.append(
                {"date": date, "category_label": None, "text": chunk.strip()}
            )
            continue
        for label, text in sub_blocks:
            if text.strip():
                spans.append(
                    {
                        "date": date,
                        "category_label": label,
                        "text": text.strip(),
                    }
                )
    return spans


def _split_category_blocks(chunk: str) -> list[tuple[str | None, str]]:
    """チャンクを `  - {label}` 小見出し単位のブロックへ分割する。

    Returns:
        (category_label or None, block_text) のリスト。
    """
    blocks: list[tuple[str | None, str]] = []
    current_label: str | None = None
    current_lines: list[str] = []

    for line in chunk.split("\n"):
        if _DATE_LINE.match(line):
            continue  # 日付行は除外
        cat = _CATEGORY_LINE.match(line)
        if cat:
            if current_lines:
                blocks.append((current_label, "\n".join(current_lines)))
            current_label = cat.group(1).strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append((current_label, "\n".join(current_lines)))
    return blocks


def collect(
    entry: dict, grep_index: list[dict], chunks: list[str]
) -> tuple[list[str], set[str], list[str]]:
    """セクションの routing に基づき evidence を収集する（決定論）。

    synthetic は全日付チャンクを供給する。extractive はカテゴリ全件収集
    （top-k 制限なし）＋ keyword grep（label=None スパンも対象）。

    Args:
        entry: routing のセクションエントリ（mode/categories/keywords）。
        grep_index: explode_to_spans の出力。
        chunks: 日付チャンク本文（synthetic で使用）。

    Returns:
        (収集テキストのリスト, 実際に存在したカテゴリラベルの集合,
         収集に関与した日付のソート済みリスト)。
    """
    mode = entry.get("mode", "extractive")
    if mode == "synthetic":
        syn_present = {
            s["category_label"] for s in grep_index if s["category_label"]
        }
        syn_dates = sorted({s["date"] for s in grep_index})
        return list(chunks), syn_present, syn_dates

    target_labels = resolve_category_labels(entry.get("categories", []))
    texts: list[str] = []
    present: set[str] = set()
    dates: set[str] = set()

    # カテゴリ全件収集（該当ラベルの span を全件回収）
    for span in grep_index:
        if span["category_label"] in target_labels:
            texts.append(span["text"])
            present.add(span["category_label"])
            dates.add(span["date"])

    # keyword grep（カテゴリ越境・label=None スパンを補完）
    keywords = entry.get("keywords", [])
    if keywords:
        pattern = re.compile("|".join(re.escape(k) for k in keywords))
        for span in grep_index:
            if pattern.search(span["text"]) and span["text"] not in texts:
                texts.append(span["text"])
                dates.add(span["date"])
                if span["category_label"]:
                    present.add(span["category_label"])

    return texts, present, sorted(dates)


def execute_search_tool(
    tool: str, args: dict, grep_index: list[dict]
) -> list[str]:
    """LLM が選んだ検索ツールを決定論的に実行しスパン本文を返す。

    ハイブリッド検索の補完ループ（agentic 部分）から呼ばれる。クエリ生成は
    LLM、実行（grep）は決定論。サポートするツール:
        keyword: args["keyword"] を含むスパン（部分文字列・小文字無視）。
            同義語・略語・言い換えの取りこぼし（語彙ミスマッチ）を埋める。
        category: args["category"] とラベルが一致するスパンを全件回収。
            routing に静的定義されていないカテゴリを LLM が実行時に指定して
            拾える（カテゴリ越境対策）。正規化（全半角・空白）後の完全一致を
            優先し、無ければ2文字以上で部分一致フォールバック（表記揺れ耐性）。
        date_range: args["start_date"] <= date <= args["end_date"] のスパン。

    Args:
        tool: "keyword" / "category" / "date_range"。
        args: ツール引数（keyword / category / start_date / end_date）。
        grep_index: explode_to_spans の出力。

    Returns:
        該当スパンの本文リスト（該当なし・未知ツールは空）。
    """
    if tool == "keyword":
        keyword = str(args.get("keyword", "")).strip()
        if not keyword:
            return []
        low = keyword.lower()
        return [s["text"] for s in grep_index if low in s["text"].lower()]
    if tool == "category":
        category = str(args.get("category", "")).strip()
        if not category:
            return []
        norm_q = _normalize_label(category)
        # 正規化後の完全一致（全半角・空白の揺れを吸収）を優先。
        exact = [
            s["text"]
            for s in grep_index
            if s.get("category_label")
            and _normalize_label(s["category_label"]) == norm_q
        ]
        if exact:
            return exact
        # 部分一致フォールバック（例: 「薬剤」→「薬剤・服薬」）。誤マッチ
        # 回避のため2文字以上に限定し、いずれかが他方を包含する場合のみ。
        if len(norm_q) < 2:
            return []
        return [
            s["text"]
            for s in grep_index
            if s.get("category_label")
            and (
                norm_q in _normalize_label(s["category_label"])
                or _normalize_label(s["category_label"]) in norm_q
            )
        ]
    if tool == "date_range":
        start = str(args.get("start_date", "")).strip()
        end = str(args.get("end_date", "")).strip()
        if not start or not end:
            return []
        return [
            s["text"]
            for s in grep_index
            if s.get("date") and start <= s["date"] <= end
        ]
    return []


def absent_categories(entry: dict, present_labels: set[str]) -> list[str]:
    """routing が要求するが記録に存在しないカテゴリラベルを返す。

    記録自体に無いカテゴリは refill しても得られないため、
    「記録なし（要確認）」として明示・review 対象にする。

    Args:
        entry: routing のセクションエントリ。
        present_labels: collect で実際に見つかったラベル集合。

    Returns:
        欠落カテゴリラベルのソート済みリスト。
    """
    target = resolve_category_labels(entry.get("categories", []))
    return sorted(target - present_labels)
