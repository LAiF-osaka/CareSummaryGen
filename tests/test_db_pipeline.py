"""DB→context 一気通貫テスト（in-memory SQLite、LLM不要）。

SqlRecordSource → normalize → sample → phi_mask → render → input_adapter の
全段が接続されることを SQLite で検証する。
"""

from sqlalchemy import create_engine, text

from adapters.normalizer import normalize
from adapters.phi_masker import mask
from adapters.pipeline import build_context_from_db
from adapters.sampler import sample
from adapters.sql_source import SqlRecordSource
from graph.nodes.ingest import _build_search_index, _extract_summary_header
from query_specs_loader.loader import load_query_spec


def _sqlite_engine():
    """sql_sample のスキーマ・データを持つ in-memory SQLite を作る。"""
    engine = create_engine("sqlite://")
    with engine.begin() as conn:
        conn.execute(
            text(
                "CREATE TABLE vital_signs (patient_id TEXT, encounter_id TEXT,"
                " measured_at TEXT, item_code TEXT, value TEXT, unit TEXT,"
                " recorder_name TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE nursing_records (patient_id TEXT,"
                " encounter_id TEXT, recorded_at TEXT, soap_s TEXT,"
                " soap_o TEXT, soap_a TEXT, soap_p TEXT, recorder_name TEXT)"
            )
        )
        conn.execute(
            text(
                "CREATE TABLE infections (patient_id TEXT, encounter_id TEXT,"
                " name TEXT, detail TEXT)"
            )
        )
        conn.execute(
            text(
                "INSERT INTO vital_signs VALUES"
                " ('P1','E1','2023-02-09','31001368','37.8','℃','看護師A'),"
                " ('P1','E1','2023-02-15','31000001','98','%','看護師B')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO nursing_records VALUES"
                " ('P1','E1','2023-02-09','息苦しい','湿性ラ音','悪化リスク',"
                "'体位ドレナージ','看護師A')"
            )
        )
        conn.execute(
            text(
                "INSERT INTO infections VALUES"
                " ('P1','E1','MRSA','接触予防継続。連絡先 03-1234-5678')"
            )
        )
    return engine


def test_sql_source_fetch():
    """SqlRecordSource が record_category ごとに行を返すこと。"""
    spec = load_query_spec("sql_sample")
    src = SqlRecordSource(engine=_sqlite_engine())
    raw = src.fetch("P1", "E1", spec)
    assert len(raw["vital_sign"]) == 2
    assert len(raw["nursing_note"]) == 1
    assert raw["vital_sign"][0]["item_code"] == "31001368"


def test_normalize_resolves_codes():
    """normalize が item コードを名称解決し、感染症を cross_cutting にすること。"""
    spec = load_query_spec("sql_sample")
    src = SqlRecordSource(engine=_sqlite_engine())
    raw = src.fetch("P1", "E1", spec)
    rs = normalize(raw, spec, "P1", "E1")
    # コード解決: 31001368 → 体温
    vital = [r for r in rs.records if r.category.value == "vital_sign"]
    labels = {f.label for r in vital for f in r.fields}
    assert "体温" in labels
    assert "SpO2" in labels
    # 感染症は cross_cutting
    inf = [r for r in rs.records if r.category.value == "infection"]
    assert inf and inf[0].cross_cutting


def test_phi_mask_text_fields():
    """phi_masker が自由記述の電話番号をマスクすること。"""
    spec = load_query_spec("sql_sample")
    src = SqlRecordSource(engine=_sqlite_engine())
    rs = normalize(src.fetch("P1", "E1", spec), spec, "P1", "E1")
    rs = mask(rs)
    all_text = "\n".join(f.value or "" for r in rs.records for f in r.fields)
    assert "03-1234-5678" not in all_text
    assert "[電話番号]" in all_text


def test_build_context_from_db_end_to_end():
    """DB→context が日付チャンク Markdown を生成し input_adapter で分割できること。"""
    engine = _sqlite_engine()
    context = build_context_from_db(
        "P1", "E1", "sql_sample", source_adapter=SqlRecordSource(engine=engine)
    )
    assert context.startswith("# 患者ID: P1")
    assert "- 20230209" in context
    assert "体温: 37.8℃" in context
    # 感染症（cross_cutting）はサマリヘッダへ
    header = _extract_summary_header(context)
    assert "MRSA" in header
    # input_adapter で日付分割
    chunks, index = _build_search_index(context)
    dates = [c["date"] for c in index]
    assert "20230209" in dates and "20230215" in dates


def test_sampler_keeps_records_under_threshold():
    """4件以下の項目は間引かれないこと。"""
    spec = load_query_spec("sql_sample")
    rs = normalize(
        SqlRecordSource(engine=_sqlite_engine()).fetch("P1", "E1", spec),
        spec,
        "P1",
        "E1",
    )
    before = len(rs.records)
    after = len(sample(rs, spec).records)
    assert after == before  # 各項目1件のみ
