"""query_spec の Pydantic モデル。

DB取得仕様を「論理層（record_category / columns(role) / contributes_to /
sampling）」と「retrieval層（source_type 固有: sql 等）」に分離して保持する。

詳細設計: docs/db-input-design.md §2 を参照。
"""

from pydantic import BaseModel, Field


class ColumnSpec(BaseModel):
    """列の意味役割定義（取得元非依存）。

    Attributes:
        role: 意味役割（datetime/item/value/unit/text/subtype/recorder/phi/id）。
        label: 表示名（任意）。
        codesystem: コード解決に使うコード体系ID（role=item 時に任意）。
        datetime_kind: 日付の意味（recorded/measured/performed、role=datetime 時）。
    """

    role: str
    label: str | None = None
    codesystem: str | None = None
    datetime_kind: str | None = None


class RecordSpec(BaseModel):
    """1記録区分の取得・正規化定義。"""

    record_category: str
    contributes_to: list[str] = Field(default_factory=list)
    columns: dict[str, ColumnSpec]
    sampling: dict | None = None
    retrieval: dict


class QuerySpec(BaseModel):
    """取得仕様全体。"""

    spec_id: str
    source_type: str
    keys: dict[str, dict] = Field(default_factory=dict)
    records: list[RecordSpec]
