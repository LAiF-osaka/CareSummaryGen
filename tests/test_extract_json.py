"""llm.client.extract_json のテスト。

gpt-oss が JSON に reasoning trace やコードフェンスを混ぜて返すケースを
堅牢にパースできることを検証する。
"""

from llm.client import _first_balanced_object, extract_json


def test_extract_json_plain():
    """素の JSON をパースできること。"""
    assert extract_json('{"queries": ["a", "b"]}') == {"queries": ["a", "b"]}


def test_extract_json_with_code_fence():
    """```json フェンス内の JSON を抽出できること。"""
    text = '説明文\n```json\n{"body": "バイタル", "x": 1}\n```\nおわり'
    assert extract_json(text) == {"body": "バイタル", "x": 1}


def test_extract_json_with_reasoning_prefix():
    """reasoning trace の後に続く JSON を抽出できること。"""
    text = (
        "まず対象セクションを分析します。次が結果です。\n"
        '{"body": "服薬指導", "cited_dates": ["20240101"]}'
    )
    result = extract_json(text)
    assert result["body"] == "服薬指導"
    assert result["cited_dates"] == ["20240101"]


def test_extract_json_with_trailing_text():
    """JSON の後ろに説明文が続いても抽出できること。"""
    text = '{"body": "体温"}\n\n以上です。'
    assert extract_json(text) == {"body": "体温"}


def test_extract_json_with_braces_in_string():
    """文字列値内に波括弧があっても平衡を正しく判定すること。"""
    text = '{"note": "use {x} here", "body": "y"}'
    result = extract_json(text)
    assert result["body"] == "y"
    assert result["note"] == "use {x} here"


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
