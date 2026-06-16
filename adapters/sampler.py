"""時系列サンプリング（トークン量制御）。

record_category の sampling.strategy が "extremes" の場合、同一項目の
時系列を first/last（日付）と min/max（数値）に絞る。異常方向の外れ値を
落とさないため平均ではなく極値を採用する。Normalizer と Renderer の間の
独立ステップ（単一責任）。

詳細設計: docs/db-input-design.md §3.2 を参照。
"""

import re

from adapters.models import ClinicalRecord, NormalizedRecordSet
from query_specs_loader.models import QuerySpec

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def sample(
    record_set: NormalizedRecordSet, spec: QuerySpec
) -> NormalizedRecordSet:
    """extremes 指定カテゴリの時系列を極値へ間引く。

    Args:
        record_set: 正規化済みレコード集合。
        spec: query_spec（sampling 設定の参照元）。

    Returns:
        間引き後の新しい NormalizedRecordSet。
    """
    extremes_cats = {
        r.record_category
        for r in spec.records
        if (r.sampling or {}).get("strategy") == "extremes"
    }
    if not extremes_cats:
        return record_set

    kept: list[ClinicalRecord] = []
    # 間引き対象カテゴリのレコードを項目ラベル単位でまとめる
    by_item: dict[tuple[str, str], list[ClinicalRecord]] = {}
    for record in record_set.records:
        if record.category.value not in extremes_cats:
            kept.append(record)
            continue
        item_label = record.fields[0].label if record.fields else ""
        by_item.setdefault((record.category.value, item_label), []).append(
            record
        )

    for records in by_item.values():
        kept.extend(_extremes(records))

    return record_set.model_copy(update={"records": kept})


def _extremes(records: list[ClinicalRecord]) -> list[ClinicalRecord]:
    """同一項目のレコード群から first/last/min/max を選ぶ（重複除去）。"""
    if len(records) <= 4:
        return records

    dated = [r for r in records if r.event_date is not None]
    if not dated:
        return records[:4]

    by_date = sorted(dated, key=lambda r: r.event_date)
    selected = {id(by_date[0]), id(by_date[-1])}
    picks = [by_date[0], by_date[-1]]

    # 数値の min/max
    numeric = [(r, _numeric(r)) for r in dated]
    numeric = [(r, v) for r, v in numeric if v is not None]
    if numeric:
        lo = min(numeric, key=lambda x: x[1])[0]
        hi = max(numeric, key=lambda x: x[1])[0]
        for r in (lo, hi):
            if id(r) not in selected:
                selected.add(id(r))
                picks.append(r)

    return picks


def _numeric(record: ClinicalRecord) -> float | None:
    """レコードの先頭フィールド値から数値を抽出する。"""
    if not record.fields or record.fields[0].value is None:
        return None
    match = _NUM.search(record.fields[0].value)
    return float(match.group()) if match else None
