"""search ノードの検索ツールのテスト（LLM不要）。"""

from graph.nodes.search import (
    _execute_date_range_search,
    _execute_keyword_search,
    _execute_tool_call,
)


def test_keyword_search_found():
    """キーワードが含まれるチャンクが返ること。"""
    chunks = ["BP 120/80 体温 36.5", "投薬なし", "SpO2 98% BP 130/85"]
    results = _execute_keyword_search("BP", chunks)
    assert len(results) == 2
    assert "BP 120/80" in results[0]


def test_keyword_search_case_insensitive():
    """大文字小文字を区別しないこと。"""
    chunks = ["spo2 98%", "SPO2 低下", "血圧正常"]
    results = _execute_keyword_search("SpO2", chunks)
    assert len(results) == 2


def test_keyword_search_not_found():
    """キーワードがない場合は空リストを返すこと。"""
    chunks = ["入院時の状態", "退院準備"]
    results = _execute_keyword_search("手術", chunks)
    assert len(results) == 0


def test_date_range_search():
    """日付範囲でチャンクが返ること。"""
    chunks = ["記録A", "記録B", "記録C"]
    index = [
        {"date": "20230209"},
        {"date": "20230210"},
        {"date": "20230215"},
    ]
    results = _execute_date_range_search("20230209", "20230210", chunks, index)
    assert len(results) == 2
    assert "記録A" in results
    assert "記録B" in results


def test_execute_tool_call_keyword():
    """_execute_tool_call がキーワード検索を実行すること。"""
    chunks = ["バイタル BP 120", "投薬記録"]
    index = [{"date": "20230209"}, {"date": "20230210"}]
    results = _execute_tool_call(
        "search_by_keyword", {"keyword": "バイタル"}, chunks, index
    )
    assert len(results) == 1


def test_execute_tool_call_date_range():
    """_execute_tool_call が日付範囲検索を実行すること。"""
    chunks = ["記録1", "記録2"]
    index = [{"date": "20230101"}, {"date": "20230201"}]
    results = _execute_tool_call(
        "search_by_date_range",
        {"start_date": "20230101", "end_date": "20230131"},
        chunks,
        index,
    )
    assert len(results) == 1
