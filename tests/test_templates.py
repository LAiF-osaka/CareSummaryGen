"""テンプレート管理のテスト。"""

import pytest

from templates_loader.loader import (
    build_format_instruction,
    list_templates,
    load_template,
)


def test_load_template_hanwa():
    """阪和病院テンプレートがロードできること。"""
    t = load_template("hanwa")
    assert t["id"] == "hanwa"
    assert t["name"] == "阪和病院"
    assert len(t["sections"]) == 6
    assert t["sections"][0]["key"] == "instruction"


def test_load_template_shinkinen():
    """新記念病院テンプレートがロードできること。"""
    t = load_template("shinkinen")
    assert t["id"] == "shinkinen"
    assert len(t["sections"]) == 2


def test_load_template_not_found():
    """存在しないテンプレートで FileNotFoundError が発生すること。"""
    with pytest.raises(FileNotFoundError):
        load_template("nonexistent")


def test_list_templates():
    """テンプレート一覧が取得できること。"""
    templates = list_templates()
    assert len(templates) >= 2
    ids = [t["id"] for t in templates]
    assert "hanwa" in ids
    assert "shinkinen" in ids


def test_build_format_instruction():
    """フォーマット指示文が生成できること。"""
    t = load_template("hanwa")
    fmt = build_format_instruction(t)
    assert "--- 指導した内容 ---" in fmt
    assert "--- 継続される問題（今後のリスク） ---" in fmt
