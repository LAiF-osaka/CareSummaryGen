"""grep 索引・収集・欠落検出のテスト（v2、LLM不要）。"""

from graph.search_index import absent_categories, collect, explode_to_spans


def _db_chunks():
    """DB renderer 形式（カテゴリ小見出しあり）の2チャンク。"""
    chunks = [
        "- 20230209\n  - バイタルサイン\n    体温: 37.8℃\n    SpO2: 94%\n"
        "  - 看護記録\n    S: 息苦しい",
        "- 20230215\n  - バイタルサイン\n    体温: 36.6℃",
    ]
    index = [{"date": "20230209"}, {"date": "20230215"}]
    return chunks, index


def test_explode_db_chunks_by_category():
    """カテゴリ小見出しでスパンに分解されること。"""
    chunks, index = _db_chunks()
    spans = explode_to_spans(chunks, index)
    labels = {s["category_label"] for s in spans}
    assert "バイタルサイン" in labels
    assert "看護記録" in labels
    # 体温がバイタルサインのスパンに含まれる
    vital = [s for s in spans if s["category_label"] == "バイタルサイン"]
    assert any("37.8℃" in s["text"] for s in vital)


def test_explode_non_db_chunk_label_none():
    """小見出しの無いチャンクは label=None の単一スパンになること。"""
    chunks = ["- 20230209\n  カルテ#1\n    自由記述の経過"]
    index = [{"date": "20230209"}]
    spans = explode_to_spans(chunks, index)
    assert len(spans) == 1
    assert spans[0]["category_label"] is None
    assert "自由記述" in spans[0]["text"]


def test_collect_extractive_category_exhaustive():
    """extractive: カテゴリ全件収集で全該当スパンを収集すること。"""
    chunks, index = _db_chunks()
    spans = explode_to_spans(chunks, index)
    entry = {
        "mode": "extractive",
        "categories": ["vital_sign"],  # → ラベル「バイタルサイン」
        "keywords": [],
    }
    texts, present, dates = collect(entry, spans, chunks)
    # 2日分のバイタルが両方収集される（top-k で切られない）
    assert any("37.8℃" in t for t in texts)
    assert any("36.6℃" in t for t in texts)
    assert "バイタルサイン" in present
    assert "20230209" in dates and "20230215" in dates


def test_collect_keyword_picks_label_none():
    """keyword grep が label=None スパンも拾うこと。"""
    chunks = ["- 20230209\n  カルテ#1\n    退院指導を実施"]
    index = [{"date": "20230209"}]
    spans = explode_to_spans(chunks, index)
    entry = {
        "mode": "extractive",
        "categories": ["procedure"],
        "keywords": ["指導"],
    }
    texts, _present, _dates = collect(entry, spans, chunks)
    assert any("退院指導" in t for t in texts)


def test_collect_synthetic_returns_all_chunks():
    """synthetic: 全日付チャンクを供給すること。"""
    chunks, index = _db_chunks()
    spans = explode_to_spans(chunks, index)
    entry = {
        "mode": "synthetic",
        "categories": ["nursing_note"],
        "keywords": [],
    }
    texts, _present, _dates = collect(entry, spans, chunks)
    assert len(texts) == 2  # 全チャンク


def test_absent_categories_flags_missing():
    """記録に存在しないカテゴリを欠落として返すこと。"""
    chunks, index = _db_chunks()
    spans = explode_to_spans(chunks, index)
    entry = {
        "mode": "extractive",
        "categories": ["vital_sign", "medication"],  # medication は記録に無い
        "keywords": [],
    }
    _texts, present, _dates = collect(entry, spans, chunks)
    absent = absent_categories(entry, present)
    assert "薬剤・服薬" in absent  # medication のラベル
    assert "バイタルサイン" not in absent
