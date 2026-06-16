"""PHI（個人健康情報）マスク。

構造化列の除外は Normalizer（role=phi）で行う。本モジュールは自由記述
（is_text）に埋め込まれた電話番号・郵便番号等を正規表現で非可逆マスクする。
対応表を作らない（一貫トークンにしない）ことで漏洩経路を断つ。

氏名・施設名の NER マスクは依存追加が必要なため将来オプション
（docs/db-input-design.md §7.2）。本実装は決定論的な regex マスクを提供する。
"""

import re

from adapters.models import ClinicalRecord, NormalizedRecordSet, RecordField

# 電話番号（ハイフン区切り）・郵便番号・長い数字列
_PATTERNS = [
    (re.compile(r"\b0\d{1,4}-\d{1,4}-\d{3,4}\b"), "[電話番号]"),
    (re.compile(r"\b\d{3}-\d{4}\b"), "[郵便番号]"),
    (re.compile(r"\b\d{10,}\b"), "[数字列]"),
    (re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"), "[メールアドレス]"),
]


def mask(record_set: NormalizedRecordSet) -> NormalizedRecordSet:
    """自由記述フィールドの PHI パターンを非可逆マスクする。

    Args:
        record_set: 正規化済みレコード集合。

    Returns:
        マスク後の新しい NormalizedRecordSet。
    """
    masked_records: list[ClinicalRecord] = []
    for record in record_set.records:
        masked_fields = [_mask_field(field) for field in record.fields]
        masked_records.append(
            record.model_copy(update={"fields": masked_fields})
        )
    return record_set.model_copy(update={"records": masked_records})


def _mask_field(field: RecordField) -> RecordField:
    """自由記述フィールドの値をマスクする（構造化値は対象外）。"""
    if not field.is_text or not field.value:
        return field
    value = field.value
    for pattern, replacement in _PATTERNS:
        value = pattern.sub(replacement, value)
    return field.model_copy(update={"value": value})
