"""routing 定義ローダー（v2 agentic search）。

`templates/<id>.routing.yaml` を読み込み、各セクションの
記録カテゴリ・キーワード・モードを返す。categories が
RecordCategory の有効なメンバであることを検証する。

詳細設計: docs/agentic-search-redesign.md §9 を参照。
"""

from pathlib import Path

import yaml

from adapters.models import CATEGORY_LABELS, RecordCategory

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"

# 有効な mode
_VALID_MODES = {"extractive", "synthetic"}


def load_routing(template_id: str) -> dict:
    """routing YAML をロードし検証して返す。

    Args:
        template_id: テンプレート識別子（例: "hanwa"）。

    Returns:
        section_key -> {mode, categories, keywords, required_items} の辞書。

    Raises:
        FileNotFoundError: routing ファイルが存在しない場合。
        ValueError: categories が RecordCategory に存在しない、
            または mode が不正な場合。
    """
    path = TEMPLATE_DIR / f"{template_id}.routing.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"routing 定義 '{template_id}' が見つかりません: {path}"
        )

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    sections = data.get("sections", {})
    validated: dict = {}
    for key, entry in sections.items():
        mode = entry.get("mode", "extractive")
        if mode not in _VALID_MODES:
            raise ValueError(
                f"routing '{template_id}' の section '{key}' の mode が不正: {mode}"
            )
        categories = entry.get("categories", []) or []
        for cat in categories:
            # RecordCategory メンバかつ CATEGORY_LABELS にキーが存在すること
            try:
                member = RecordCategory(cat)
            except ValueError as exc:
                raise ValueError(
                    f"routing '{template_id}' section '{key}': "
                    f"未知のカテゴリ '{cat}'"
                ) from exc
            if member not in CATEGORY_LABELS:
                raise ValueError(
                    f"CATEGORY_LABELS に '{cat}' のラベルがありません"
                )
        validated[key] = {
            "mode": mode,
            "categories": list(categories),
            "keywords": list(entry.get("keywords", []) or []),
            "required_items": list(entry.get("required_items", []) or []),
        }

    return validated


def resolve_category_labels(categories: list[str]) -> set[str]:
    """カテゴリ enum 値のリストを日本語ラベルの集合へ解決する。

    grep のアンカーは CATEGORY_LABELS の日本語ラベルであるため、
    routing の enum 値を grep 前にラベルへ変換する（単一写像）。

    Args:
        categories: RecordCategory の enum 値のリスト。

    Returns:
        対応する日本語ラベルの集合（未知の値は無視）。
    """
    labels: set[str] = set()
    for cat in categories:
        try:
            labels.add(CATEGORY_LABELS[RecordCategory(cat)])
        except (ValueError, KeyError):
            continue
    return labels
