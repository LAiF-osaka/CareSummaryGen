"""生レコードの正規化（取得元固有 → ClinicalRecord）。

query_spec の columns(role) に従って各行を ClinicalRecord へ変換する。
role=item かつ codesystem 指定の列はコード→名称解決する。
患者横断カテゴリには cross_cutting を立て、サマリヘッダ領域へ回す。
取得0件の record_category は missing_categories に記録する。

詳細設計: docs/db-input-design.md §3 を参照。
"""

from datetime import datetime

from adapters.models import (
    ClinicalRecord,
    NormalizedRecordSet,
    RecordCategory,
    RecordField,
)
from query_specs_loader.loader import load_codesystem
from query_specs_loader.models import QuerySpec, RecordSpec

# 日付に紐づかない患者横断カテゴリ（サマリヘッダ領域へ）
_CROSS_CUTTING = {
    RecordCategory.PATIENT_PROFILE,
    RecordCategory.ALLERGY,
    RecordCategory.INFECTION,
    RecordCategory.NURSING_PROBLEM,
    RecordCategory.NURSING_ACUITY,
}

_UNRESOLVED = "[未解決コード:{code}]"


def normalize(
    raw: dict[str, list[dict]],
    spec: QuerySpec,
    patient_id: str,
    encounter_id: str,
) -> NormalizedRecordSet:
    """生レコード集合を NormalizedRecordSet へ正規化する。

    Args:
        raw: {record_category: [row_dict]}（取得元固有スキーマ）。
        spec: query_spec。
        patient_id: 患者ID。
        encounter_id: 入院ID。

    Returns:
        正規化済みレコード集合。
    """
    spec_by_cat = {r.record_category: r for r in spec.records}
    codesystem_cache: dict[str, dict[str, str]] = {}

    records: list[ClinicalRecord] = []
    missing: list[RecordCategory] = []

    for category, rows in raw.items():
        try:
            cat_enum = RecordCategory(category)
        except ValueError:
            continue
        if not rows:
            missing.append(cat_enum)
            continue
        rspec = spec_by_cat.get(category)
        if rspec is None:
            continue
        for row in rows:
            records.append(
                _normalize_row(row, rspec, cat_enum, codesystem_cache)
            )

    return NormalizedRecordSet(
        patient_id=patient_id,
        encounter_id=encounter_id,
        records=records,
        missing_categories=missing,
    )


def _normalize_row(
    row: dict,
    rspec: RecordSpec,
    category: RecordCategory,
    codesystem_cache: dict[str, dict[str, str]],
) -> ClinicalRecord:
    """1行を ClinicalRecord へ変換する。"""
    columns = rspec.columns

    event_date: datetime | None = None
    date_kind: str | None = None
    recorder: str | None = None
    subtype: str | None = None
    # codesystem で解決された項目名（バイタル等。value/unit と1フィールドに統合）
    code_item_label: str | None = None
    item_value: str | None = None
    item_unit: str | None = None
    # codesystem 無しの item（セルが値・label がフィールド名）と text
    direct_fields: list[RecordField] = []
    text_fields: list[RecordField] = []

    for col_name, col_spec in columns.items():
        if col_name not in row:
            continue
        cell = row[col_name]
        role = col_spec.role

        if role == "datetime":
            event_date = _parse_date(cell)
            date_kind = col_spec.datetime_kind
        elif role == "recorder":
            recorder = _as_str(cell)
        elif role == "subtype":
            subtype = _as_str(cell)
        elif role == "phi":
            continue  # PHI 列は出力しない（§7）
        elif role == "item":
            if col_spec.codesystem:
                code_item_label = _resolve_code(
                    cell, col_spec, codesystem_cache
                )
            else:
                # セルが値、col_spec.label がフィールド名（例: 感染症=MRSA）
                val = _as_str(cell)
                if val:
                    direct_fields.append(
                        RecordField(
                            label=col_spec.label or col_name, value=val
                        )
                    )
        elif role == "value":
            item_value = _as_str(cell)
        elif role == "unit":
            item_unit = _as_str(cell)
        elif role == "text":
            val = _as_str(cell)
            if val:
                text_fields.append(
                    RecordField(
                        label=col_spec.label or col_name,
                        value=val,
                        is_text=True,
                    )
                )
        elif role in ("id", "link"):
            continue

    fields: list[RecordField] = []
    if code_item_label is not None:
        # codesystem 解決済み項目 + value + unit を1フィールドに統合
        fields.append(
            RecordField(
                label=code_item_label, value=item_value, unit=item_unit
            )
        )
    elif item_value is not None:
        fields.append(
            RecordField(label="値", value=item_value, unit=item_unit)
        )
    fields.extend(direct_fields)
    fields.extend(text_fields)

    return ClinicalRecord(
        event_date=event_date,
        date_kind=date_kind,
        category=category,
        subtype=subtype,
        fields=fields,
        recorder=recorder,
        cross_cutting=category in _CROSS_CUTTING,
    )


def _resolve_code(
    cell, col_spec, codesystem_cache: dict[str, dict[str, str]]
) -> str:
    """role=item かつ codesystem 指定のコードを名称解決する。

    解決できないコードは欠損と区別するため [未解決コード:...] で残す。
    """
    raw_value = _as_str(cell) or ""
    cs_id = col_spec.codesystem
    if cs_id not in codesystem_cache:
        codesystem_cache[cs_id] = load_codesystem(cs_id)
    mapping = codesystem_cache[cs_id]
    if raw_value in mapping:
        return mapping[raw_value]
    return _UNRESOLVED.format(code=raw_value)


def _parse_date(cell) -> datetime | None:
    """セル値を datetime へ変換する（複数形式に対応）。"""
    if cell is None:
        return None
    if isinstance(cell, datetime):
        return cell
    text_value = str(cell).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(text_value[: len(fmt) + 4], fmt)
        except ValueError:
            continue
    # ISO 形式の保険
    try:
        return datetime.fromisoformat(text_value)
    except ValueError:
        return None


def _as_str(cell) -> str | None:
    """セル値を文字列へ（None はそのまま）。"""
    if cell is None:
        return None
    return str(cell)
