"""routing ローダーと契約（routing↔RecordCategory 整合）のテスト。"""

import pytest

from adapters.models import CATEGORY_LABELS, RecordCategory
from templates_loader.routing import load_routing, resolve_category_labels


def test_load_hanwa_routing():
    """hanwa の routing がロードできること。"""
    r = load_routing("hanwa")
    assert set(r.keys()) == {
        "instruction",
        "medical_equipment",
        "nursing_process",
        "patient_condition",
        "risks",
        "others",
    }
    assert r["nursing_process"]["mode"] == "synthetic"
    assert r["instruction"]["mode"] == "extractive"


def test_load_shinkinen_routing():
    """shinkinen の routing がロードできること。"""
    r = load_routing("shinkinen")
    assert set(r.keys()) == {"progress", "remarks"}


def test_routing_not_found():
    """存在しない routing で FileNotFoundError。"""
    with pytest.raises(FileNotFoundError):
        load_routing("nonexistent")


@pytest.mark.parametrize("template_id", ["hanwa", "shinkinen"])
def test_routing_categories_are_valid_record_categories(template_id):
    """契約: 全 routing の categories が RecordCategory かつラベル定義あり。"""
    r = load_routing(template_id)
    for key, entry in r.items():
        for cat in entry["categories"]:
            member = RecordCategory(cat)  # 不正なら ValueError
            assert member in CATEGORY_LABELS, f"{cat} にラベルなし ({key})"


def test_resolve_category_labels():
    """enum 値が日本語ラベルへ解決されること。"""
    labels = resolve_category_labels(["vital_sign", "procedure"])
    assert "バイタルサイン" in labels
    assert "処置・医療機器" in labels


def test_resolve_unknown_category_ignored():
    """未知のカテゴリは無視されること（例外を投げない）。"""
    labels = resolve_category_labels(["vital_sign", "bogus"])
    assert labels == {"バイタルサイン"}
