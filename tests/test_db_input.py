"""DB入力アダプタ（Phase 1）の接続テスト。

NormalizedRecordSet → build_context（Markdown化）→ input_adapter の
日付分割・サマリヘッダ保持が正しく接続されることを検証する。
LLM 不要。
"""

from datetime import datetime

from adapters.models import (
    ClinicalRecord,
    NormalizedRecordSet,
    RecordCategory,
    RecordField,
)
from adapters.pipeline import build_context
from graph.nodes.input_adapter import (
    _build_search_index,
    _extract_summary_header,
)


def _sample_record_set() -> NormalizedRecordSet:
    """肺炎入院の最小サンプル（患者横断情報＋2日分の経過＋欠損）。"""
    return NormalizedRecordSet(
        patient_id="DEMO001",
        encounter_id="ENC001",
        missing_categories=[RecordCategory.LAB_RESULT],
        records=[
            # 患者横断情報（cross_cutting）→ サマリヘッダ領域
            ClinicalRecord(
                category=RecordCategory.INFECTION,
                cross_cutting=True,
                fields=[RecordField(label="感染症", value="MRSA 陽性")],
            ),
            ClinicalRecord(
                category=RecordCategory.NURSING_PROBLEM,
                cross_cutting=True,
                subtype="転帰: 継続",
                fields=[
                    RecordField(label="看護問題", value="#1 ガス交換障害"),
                    RecordField(
                        label="目標", value="SpO2 95%以上を維持", is_text=True
                    ),
                ],
            ),
            # 日付チャンク（2日分）
            ClinicalRecord(
                event_date=datetime(2023, 2, 9),
                date_kind="measured",
                category=RecordCategory.VITAL_SIGN,
                recorder="看護師A",
                fields=[
                    RecordField(label="体温", value="37.8", unit="℃"),
                    RecordField(label="SpO2", value="94", unit="%"),
                ],
            ),
            ClinicalRecord(
                event_date=datetime(2023, 2, 15),
                date_kind="measured",
                category=RecordCategory.VITAL_SIGN,
                recorder="看護師B",
                fields=[
                    RecordField(label="体温", value="36.6", unit="℃"),
                    RecordField(label="SpO2", value="98", unit="%"),
                ],
            ),
        ],
    )


def test_build_context_produces_markdown():
    """build_context が患者IDヘッダと日付行を含む Markdown を生成すること。"""
    md = build_context(_sample_record_set())
    assert md.startswith("# 患者ID: DEMO001")
    assert "- 20230209" in md
    assert "- 20230215" in md
    # 擬似日付 00000000 は使わない
    assert "00000000" not in md


def test_summary_header_contains_cross_cutting():
    """患者横断情報・欠損がサマリヘッダ領域に出力されること。"""
    md = build_context(_sample_record_set())
    assert "## サマリ基本情報（全期間共通）" in md
    assert "MRSA 陽性" in md
    assert "#1 ガス交換障害" in md
    # 欠損明示
    assert "記録なし: 検査結果" in md


def test_input_adapter_splits_date_chunks():
    """生成 Markdown が input_adapter で日付チャンクへ分割されること。"""
    md = build_context(_sample_record_set())
    chunks, chunk_index = _build_search_index(md)
    dates = [c["date"] for c in chunk_index]
    assert "20230209" in dates
    assert "20230215" in dates
    # バイタル数値がチャンク本文に保持される
    joined = "\n".join(chunks)
    assert "37.8℃" in joined
    assert "94%" in joined


def test_summary_header_preserved_by_input_adapter():
    """サマリヘッダが input_adapter で抽出・保持されること（破棄されない）。"""
    md = build_context(_sample_record_set())
    header = _extract_summary_header(md)
    # 患者IDヘッダ行は除外される
    assert "# 患者ID" not in header
    # 患者横断情報は保持される
    assert "MRSA 陽性" in header
    assert "#1 ガス交換障害" in header


def test_date_chunks_exclude_cross_cutting():
    """日付チャンク本文に患者横断情報が混入しないこと。"""
    md = build_context(_sample_record_set())
    chunks, chunk_index = _build_search_index(md)
    # 日付チャンク（サマリヘッダを含む先頭チャンクを除く）に MRSA が現れない
    for chunk, meta in zip(chunks, chunk_index):
        if meta["date"] in ("20230209", "20230215"):
            assert "MRSA" not in chunk
