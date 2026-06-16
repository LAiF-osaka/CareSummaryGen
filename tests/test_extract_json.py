"""llm.client.extract_json と plan フォールバックのテスト。

gpt-oss が JSON に reasoning trace やコードフェンスを混ぜて返すケースを
堅牢にパースできることを検証する。
"""

from graph.nodes.plan import _extract_keywords_from_text
from llm.client import _first_balanced_object, extract_json

# --- extract_json ---


def test_extract_json_plain():
    """素の JSON をパースできること。"""
    assert extract_json('{"queries": ["a", "b"]}') == {"queries": ["a", "b"]}


def test_extract_json_with_code_fence():
    """```json フェンス内の JSON を抽出できること。"""
    text = '説明文\n```json\n{"queries": ["バイタル", "SpO2"]}\n```\nおわり'
    assert extract_json(text) == {"queries": ["バイタル", "SpO2"]}


def test_extract_json_with_reasoning_prefix():
    """reasoning trace の後に続く JSON を抽出できること。"""
    text = (
        "まず対象セクションを分析します。次のキーワードが適切です。\n"
        '{"queries": ["服薬指導", "退院指導"]}'
    )
    assert extract_json(text) == {"queries": ["服薬指導", "退院指導"]}


def test_extract_json_with_trailing_text():
    """JSON の後ろに説明文が続いても抽出できること。"""
    text = '{"queries": ["体温"]}\n\n以上が検索計画です。'
    assert extract_json(text) == {"queries": ["体温"]}


def test_extract_json_with_braces_in_string():
    """文字列値内に波括弧があっても平衡を正しく判定すること。"""
    text = '{"note": "use {placeholder} here", "queries": ["x"]}'
    result = extract_json(text)
    assert result["queries"] == ["x"]
    assert result["note"] == "use {placeholder} here"


def test_extract_json_failure_returns_none():
    """JSON が無いテキストでは None を返すこと。"""
    assert extract_json("これは表です | 列1 | 列2 |") is None


def test_extract_json_empty():
    """空文字では None を返すこと。"""
    assert extract_json("") is None


def test_first_balanced_object():
    """ネストした波括弧の平衡を正しく抽出すること。"""
    text = 'prefix {"a": {"b": 1}} suffix'
    assert _first_balanced_object(text) == '{"a": {"b": 1}}'


# --- plan フォールバック ---


def test_fallback_excludes_markdown_table():
    """Markdown 表のヘッダ行をキーワードに拾わないこと。"""
    text = (
        "検索計画（指導した内容セクション）\n"
        "| 検索対象 | 検索理由 |\n"
        "|---|---|\n"
        "- 服薬指導\n"
        "- 退院指導\n"
    )
    keywords = _extract_keywords_from_text(text)
    assert "服薬指導" in keywords
    assert "退院指導" in keywords
    # 表・見出し行は含まれない
    assert all("|" not in k for k in keywords)
    assert all("検索対象" not in k for k in keywords)


def test_fallback_excludes_long_sentences():
    """20文字以上の文章を除外すること。"""
    text = (
        "- これは非常に長い説明文であり検索キーワードとして不適切です\n- SpO2"
    )
    keywords = _extract_keywords_from_text(text)
    assert keywords == ["SpO2"]


def test_fallback_limits_to_five():
    """最大5件に制限すること。"""
    text = "\n".join(f"- kw{i}" for i in range(10))
    assert len(_extract_keywords_from_text(text)) == 5
