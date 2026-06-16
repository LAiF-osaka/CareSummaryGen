"""query_spec ローダーと検証のテスト。"""

import pytest

from query_specs_loader.loader import load_codesystem, load_query_spec


def test_load_sql_sample():
    """sql_sample がロード・検証できること。"""
    spec = load_query_spec("sql_sample")
    assert spec.spec_id == "sql_sample"
    assert spec.source_type == "sql"
    categories = [r.record_category for r in spec.records]
    assert "vital_sign" in categories
    assert "nursing_note" in categories


def test_load_codesystem():
    """codesystem がコード→名称で解決できること。"""
    cs = load_codesystem("medis_obs")
    assert cs["31001368"] == "体温"
    assert cs["31000001"] == "SpO2"


def test_codesystem_missing_returns_empty():
    """存在しない codesystem は空辞書を返すこと。"""
    assert load_codesystem("nonexistent") == {}


def test_invalid_bind_var_rejected(tmp_path, monkeypatch):
    """許可されないバインド変数を含む SQL を拒否すること。"""
    import query_specs_loader.loader as loader_mod

    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    (spec_dir / "bad.yaml").write_text(
        "spec_id: bad\nsource_type: sql\nrecords:\n"
        "  - record_category: vital_sign\n"
        "    columns:\n      v: {role: value}\n"
        "    retrieval:\n"
        "      sql: 'SELECT * FROM t WHERE p = :evil_param'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader_mod, "SPEC_DIR", spec_dir)
    with pytest.raises(ValueError, match="許可されないバインド変数"):
        load_query_spec("bad")


def test_invalid_category_rejected(tmp_path, monkeypatch):
    """未知の record_category を拒否すること。"""
    import query_specs_loader.loader as loader_mod

    spec_dir = tmp_path / "specs"
    spec_dir.mkdir()
    (spec_dir / "bad.yaml").write_text(
        "spec_id: bad\nsource_type: sql\nrecords:\n"
        "  - record_category: not_a_real_category\n"
        "    columns:\n      v: {role: value}\n"
        "    retrieval:\n      sql: 'SELECT 1'\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(loader_mod, "SPEC_DIR", spec_dir)
    with pytest.raises(ValueError, match="未知の record_category"):
        load_query_spec("bad")
