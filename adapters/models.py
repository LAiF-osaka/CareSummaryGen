"""DB入力アダプタの中間表現（正規化スキーマ）。

SQL / FHIR / SS-MIX2 いずれの取得元から得た行も、本モジュールの
ClinicalRecord / NormalizedRecordSet へ正規化する。MarkdownRenderer は
このモデルのみに依存し、取得元固有スキーマを知らない（疎結合の境界）。

詳細設計: docs/db-input-design.md §3 を参照。

主要クラス:
    RecordCategory: 記録区分の列挙
    RecordField: 正規化レコード内の1項目（列1つ）
    ClinicalRecord: 正規化された1記録
    NormalizedRecordSet: 1患者・1入院分のレコード集合
"""

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class RecordCategory(str, Enum):
    """記録区分。query_spec の record_category と一致させる。"""

    PATIENT_PROFILE = "patient_profile"
    ENCOUNTER = "encounter"
    VITAL_SIGN = "vital_sign"
    NURSING_NOTE = "nursing_note"
    NURSING_PROBLEM = "nursing_problem"
    MEDICATION = "medication"
    LAB_RESULT = "lab_result"
    PROCEDURE = "procedure"
    ADL = "adl"
    RISK_ASSESSMENT = "risk_assessment"
    NURSING_ACUITY = "nursing_acuity"
    ALLERGY = "allergy"
    INFECTION = "infection"
    INCIDENT = "incident"
    DISCHARGE_SUPPORT = "discharge_support"


# 記録区分 → Markdown 小見出しに用いる日本語名。
# 検索（チャンク本文の部分文字列マッチ）でヒットしやすいよう、
# テンプレートのセクション説明に現れる語彙を含める。
CATEGORY_LABELS: dict[RecordCategory, str] = {
    RecordCategory.PATIENT_PROFILE: "患者基本情報",
    RecordCategory.ENCOUNTER: "入院情報",
    RecordCategory.VITAL_SIGN: "バイタルサイン",
    RecordCategory.NURSING_NOTE: "看護記録",
    RecordCategory.NURSING_PROBLEM: "看護問題・看護計画",
    RecordCategory.MEDICATION: "薬剤・服薬",
    RecordCategory.LAB_RESULT: "検査結果",
    RecordCategory.PROCEDURE: "処置・医療機器",
    RecordCategory.ADL: "ADL・生活状況",
    RecordCategory.RISK_ASSESSMENT: "リスク評価",
    RecordCategory.NURSING_ACUITY: "看護必要度",
    RecordCategory.ALLERGY: "アレルギー",
    RecordCategory.INFECTION: "感染症",
    RecordCategory.INCIDENT: "インシデント",
    RecordCategory.DISCHARGE_SUPPORT: "退院支援・継続課題",
}


class RecordField(BaseModel):
    """正規化レコード内の1項目（列1つに対応）。

    Attributes:
        label: 表示名（コード解決後の項目名 or query_spec の label）。
        value: 値。欠損時は None。Renderer で "記録なし" に変換する。
        unit: 単位（数値項目のみ）。
        is_text: 自由記述か。True なら Markdown 本文ブロックとして展開する。
    """

    label: str
    value: str | None = None
    unit: str | None = None
    is_text: bool = False


class ClinicalRecord(BaseModel):
    """正規化された1記録。

    1行のDB結果、または1つのFHIRリソースが1 ClinicalRecord に対応する。
    Markdown化では event_date で日付チャンクへ振り分けられる。
    cross_cutting が True のレコードは日付チャンクではなく
    サマリヘッダ領域へ出力される（検索ヒットに依存させない）。

    Attributes:
        event_date: 日付軸。None は日付不明。
        date_kind: event_date の意味（recorded / measured / performed）。
        category: 記録区分。
        subtype: 記録様式区分・転帰等（任意）。
        fields: 項目群（順序保持）。
        recorder: 記録者名。
        problem_id: 看護問題ID（看護記録と看護計画の結合キー）。
        cross_cutting: 日付に紐づかない患者横断情報か（サマリヘッダ行き）。
    """

    event_date: datetime | None = None
    date_kind: str | None = None
    category: RecordCategory
    subtype: str | None = None
    fields: list[RecordField] = Field(default_factory=list)
    recorder: str | None = None
    problem_id: str | None = None
    cross_cutting: bool = False


class NormalizedRecordSet(BaseModel):
    """1患者・1入院分の正規化済みレコード集合。

    Attributes:
        patient_id: 患者ID（ローカル）。
        encounter_id: 入院ID。
        records: 全 ClinicalRecord（カテゴリ・日付混在、未ソート可）。
        missing_categories: 取得を試みたが0件だった記録区分（欠損明示用）。
    """

    patient_id: str
    encounter_id: str
    records: list[ClinicalRecord] = Field(default_factory=list)
    missing_categories: list[RecordCategory] = Field(default_factory=list)
