"""grep 索引・収集・欠落検出・検索ツールのテスト（v2、LLM不要）。"""

from graph.search_index import (
    absent_categories,
    collect,
    execute_search_tool,
    explode_to_spans,
)


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


def test_execute_search_tool_keyword():
    """補完検索の keyword ツールが該当スパン本文を返すこと。"""
    chunks, index = _db_chunks()
    spans = explode_to_spans(chunks, index)
    results = execute_search_tool("keyword", {"keyword": "息苦しい"}, spans)
    assert any("息苦しい" in r for r in results)


def test_execute_search_tool_category():
    """補完検索の category ツールがラベル一致スパンを全件返すこと。

    routing に静的定義されていないカテゴリを LLM が実行時に指定して拾える
    （カテゴリ越境対策）ことを確認する。
    """
    chunks, index = _db_chunks()
    spans = explode_to_spans(chunks, index)
    results = execute_search_tool(
        "category", {"category": "バイタルサイン"}, spans
    )
    # 2日分のバイタルサインが両方回収される
    assert any("37.8℃" in r for r in results)
    assert any("36.6℃" in r for r in results)
    # 看護記録カテゴリは含まれない
    assert not any("息苦しい" in r for r in results)


def test_execute_search_tool_category_empty():
    """category 引数が空・未知ラベルなら空を返すこと。"""
    chunks, index = _db_chunks()
    spans = explode_to_spans(chunks, index)
    assert execute_search_tool("category", {"category": ""}, spans) == []
    assert (
        execute_search_tool("category", {"category": "存在しない"}, spans)
        == []
    )


def test_execute_search_tool_category_partial_match():
    """部分一致フォールバックで表記揺れ（短縮ラベル）を吸収すること。"""
    chunks, index = _db_chunks()
    spans = explode_to_spans(chunks, index)
    # 「看護」→「看護記録」に部分一致でフォールバック
    results = execute_search_tool("category", {"category": "看護"}, spans)
    assert any("息苦しい" in r for r in results)


def test_execute_search_tool_category_normalized():
    """正規化（全半角・空白）後の完全一致で揺れを吸収すること。"""
    chunks, index = _db_chunks()
    spans = explode_to_spans(chunks, index)
    # 末尾空白・全角空白が混じってもバイタルサインに一致
    results = execute_search_tool(
        "category", {"category": "バイタル　サイン "}, spans
    )
    assert any("37.8℃" in r for r in results)


def test_execute_search_tool_date_range():
    """補完検索の date_range ツールが期間内スパンを返すこと。"""
    chunks, index = _db_chunks()
    spans = explode_to_spans(chunks, index)
    results = execute_search_tool(
        "date_range", {"start_date": "20230209", "end_date": "20230209"}, spans
    )
    assert results
    assert all("37.8℃" in r or "息苦しい" in r for r in results)


def test_execute_search_tool_unknown_returns_empty():
    """未知ツール・空引数は空リストを返すこと。"""
    chunks, index = _db_chunks()
    spans = explode_to_spans(chunks, index)
    assert execute_search_tool("bogus", {}, spans) == []
    assert execute_search_tool("keyword", {"keyword": ""}, spans) == []


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
