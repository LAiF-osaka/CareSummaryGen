"""正規化レコード集合の Markdown 化。

NormalizedRecordSet を、既存 input_adapter が期待する
「`# 患者ID:` ヘッダ → `## サマリ基本情報` → `- YYYYMMDD` 日付チャンク」
形式の Markdown へ直列化する。

詳細設計: docs/db-input-design.md §4 を参照。

設計上の不変条件:
    - 日付チャンク境界は `- YYYYMMDD` 単独行（input_adapter の正規表現契約）。
    - `- 00000000` 等の擬似日付は使わない（search の date_range 汚染を回避）。
    - 日付に紐づかない患者横断情報・欠損明示はサマリヘッダ領域に置く。
"""

from datetime import datetime

from adapters.models import (
    CATEGORY_LABELS,
    ClinicalRecord,
    NormalizedRecordSet,
)

_MISSING = "記録なし"


def render(record_set: NormalizedRecordSet) -> str:
    """正規化レコード集合を context 用 Markdown へ変換する。

    Args:
        record_set: 正規化済みレコード集合。

    Returns:
        `# 患者ID:` ヘッダ＋サマリヘッダ＋日付チャンクからなる Markdown 文字列。
    """
    lines: list[str] = [f"# 患者ID: {record_set.patient_id}", ""]

    header_lines = _render_summary_header(record_set)
    if header_lines:
        lines.append("## サマリ基本情報（全期間共通）")
        lines.extend(header_lines)
        lines.append("")

    lines.extend(_render_date_chunks(record_set))

    return "\n".join(lines).rstrip() + "\n"


def _render_summary_header(record_set: NormalizedRecordSet) -> list[str]:
    """サマリヘッダ領域（患者横断情報・日付不明・欠損）を描画する。

    日付に紐づかない情報（cross_cutting）と日付不明レコードを箇条書きで出力し、
    取得0件の記録区分を「記録なし」として明示する。
    """
    lines: list[str] = []

    # 患者横断情報（cross_cutting）と日付不明レコード
    cross = [r for r in record_set.records if r.cross_cutting]
    undated = [
        r
        for r in record_set.records
        if not r.cross_cutting and r.event_date is None
    ]
    for record in cross + undated:
        label = CATEGORY_LABELS.get(record.category, record.category.value)
        lines.append(f"- {label}")
        lines.extend(_render_fields(record, indent="    "))

    # 取得0件の記録区分を欠損として明示
    if record_set.missing_categories:
        names = "、".join(
            CATEGORY_LABELS.get(c, c.value)
            for c in record_set.missing_categories
        )
        lines.append(f"- {_MISSING}: {names}")

    return lines


def _render_date_chunks(record_set: NormalizedRecordSet) -> list[str]:
    """日付チャンク領域を描画する。

    event_date を持つ（かつ cross_cutting でない）レコードを
    `YYYYMMDD` ごとに集約し、`- YYYYMMDD` 境界で出力する。
    """
    dated = [
        r
        for r in record_set.records
        if not r.cross_cutting and r.event_date is not None
    ]
    # 日付（YYYYMMDD）昇順にソート（event_date は上のフィルタで非 None）
    dated.sort(key=lambda r: r.event_date or datetime.min)

    # 日付ごとにグループ化
    by_date: dict[str, list[ClinicalRecord]] = {}
    for record in dated:
        event_date = record.event_date
        if event_date is None:
            continue
        ymd = event_date.strftime("%Y%m%d")
        by_date.setdefault(ymd, []).append(record)

    lines: list[str] = []
    for ymd in sorted(by_date.keys()):
        lines.append(f"- {ymd}")
        # 記録区分ごとに小見出しを付けて出力
        by_category: dict[str, list[ClinicalRecord]] = {}
        for record in by_date[ymd]:
            label = CATEGORY_LABELS.get(record.category, record.category.value)
            by_category.setdefault(label, []).append(record)
        for label, records in by_category.items():
            lines.append(f"  - {label}")
            for record in records:
                lines.extend(_render_fields(record, indent="    "))
        lines.append("")

    return lines


def _render_fields(record: ClinicalRecord, indent: str) -> list[str]:
    """1レコードの項目群を Markdown 行へ変換する。

    構造化値は `{indent}{label}: {value}{unit}`、自由記述は
    `{indent}{label}:` のあとに本文を出力する。欠損は「記録なし」。
    記録者があれば末尾に注記する。

    Args:
        record: 対象レコード。
        indent: 行頭インデント（日付チャンク内は4スペース）。

    Returns:
        Markdown 行のリスト。
    """
    lines: list[str] = []

    # 転帰等の subtype を先頭に注記
    if record.subtype:
        lines.append(f"{indent}（{record.subtype}）")

    for field in record.fields:
        value = field.value if field.value not in (None, "") else _MISSING
        if field.is_text:
            lines.append(f"{indent}{field.label}:")
            for body_line in str(value).split("\n"):
                lines.append(f"{indent}  {body_line}")
        else:
            unit = field.unit or ""
            lines.append(f"{indent}{field.label}: {value}{unit}")

    if record.recorder:
        lines.append(f"{indent}（記録者: {record.recorder}）")

    return lines
