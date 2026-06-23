"""観測駆動 agentic 補完検索の受入テスト（モック LLM・実機不要）。

docs/agentic-search-redesign.md の「本物の agentic search」受入基準6項目を
決定論的に検証する。1つでも No なら「飾りの agentic」であり本物と称さない。

  1. 観測閉路    : 直前アクションの結果が次の LLM 入力に入る
  2. 再定式化    : 結果に応じてクエリが変わる・繰り返さない
  3. gap 駆動停止: 不足を同定してから停止する
  4. ガードレール: 決定論が LLM 停止を囲う（上限・進捗ゼロ）
  5. 実行時 plan : LLM が観測から探索対象を動的に決める（routing 外も）
  6. トレース    : 観測・クエリ・停止理由を search_trace に記録する
"""

from unittest.mock import patch

from graph.nodes.section_worker import (
    _coverage_report,
    _first_excerpt,
    _supplemental_search,
)
from graph.search_index import explode_to_spans

_SECTION = {"name": "看護記録", "description": "経過と観察", "key": "nursing"}
_ENTRY = {"mode": "extractive", "categories": ["nursing_note"], "keywords": []}


class _ScriptedLLM:
    """事前定義の応答列を順に返し、渡されたプロンプトを記録するモック。"""

    def __init__(self, responses: list[dict | None]):
        self.responses = list(responses)
        self.prompts: list[str] = []

    def __call__(self, prompt, schema, **kwargs):
        self.prompts.append(prompt)
        if self.responses:
            return self.responses.pop(0)
        # 応答が尽きたら充足扱いで停止させる
        return {"need_more": False, "missing_points": []}


def _spans() -> list[dict]:
    """検索対象のスパン（看護記録2件・薬剤1件）。"""
    chunks = [
        "- 20230209\n  - 看護記録\n    S: 息苦しい、酸素3L投与中",
        "- 20230210\n  - 看護記録\n    歩行時に転倒しそうになった",
        "- 20230211\n  - 薬剤・服薬\n    アムロジピン 5mg 内服開始",
    ]
    index = [{"date": "20230209"}, {"date": "20230210"}, {"date": "20230211"}]
    return explode_to_spans(chunks, index)


def _patch(scripted: _ScriptedLLM):
    return patch("graph.nodes.section_worker.chat_json", side_effect=scripted)


def test_observation_loop_is_closed():
    """基準1: 前ステップの検索結果が次の LLM 入力（観測）に入ること。"""
    scripted = _ScriptedLLM(
        [
            {
                "need_more": True,
                "missing_points": ["酸素投与の状況"],
                "tool": "keyword",
                "keyword": "酸素",
            },
            {
                "need_more": True,
                "missing_points": ["転倒リスク"],
                "tool": "keyword",
                "keyword": "転倒",
            },
            {"need_more": False, "missing_points": []},
        ]
    )
    with _patch(scripted):
        collected, _trace = _supplemental_search(
            _SECTION, _ENTRY, [], _spans()
        )

    # step1 のプロンプトに step0 の検索結果（酸素 → 追加）が観測として入る
    assert "酸素" in scripted.prompts[1]
    assert "件追加" in scripted.prompts[1]
    # カバレッジ観測（カテゴリ別件数）も含まれる
    assert "件" in scripted.prompts[1]
    # 2手の検索で両スパンが追加されている
    assert any("酸素3L" in c for c in collected)
    assert any("転倒" in c for c in collected)


def test_reformulation_zero_hit_does_not_stop():
    """基準2: ゼロ件でも即停止せず、言い換えクエリへ続行すること。"""
    scripted = _ScriptedLLM(
        [
            {
                "need_more": True,
                "missing_points": ["x"],
                "tool": "keyword",
                "keyword": "存在しない語ZZZ",  # ゼロ件
            },
            {
                "need_more": True,
                "missing_points": ["x"],
                "tool": "keyword",
                "keyword": "酸素",  # 言い換え後はヒット
            },
            {"need_more": False, "missing_points": []},
        ]
    )
    with _patch(scripted):
        collected, _trace = _supplemental_search(
            _SECTION, _ENTRY, [], _spans()
        )

    # ゼロ件後の step1 で続行し、既出クエリがプロンプトに提示される
    assert "存在しない語ZZZ" in scripted.prompts[1]
    # 言い換えクエリの結果が収集されている（即停止していない証拠）
    assert any("酸素3L" in c for c in collected)


def test_duplicate_query_detected_and_stops():
    """基準2/4: 同一クエリ繰り返しは進捗ゼロとして検出・停止すること。"""
    scripted = _ScriptedLLM(
        [
            {
                "need_more": True,
                "missing_points": ["x"],
                "tool": "keyword",
                "keyword": "存在しない語ZZZ",  # ゼロ件（no_progress=1）
            },
            {
                "need_more": True,
                "missing_points": ["x"],
                "tool": "keyword",
                "keyword": "存在しない語ZZZ",  # 重複（no_progress=2 → 停止）
            },
        ]
    )
    with _patch(scripted):
        _collected, trace = _supplemental_search(
            _SECTION, _ENTRY, [], _spans()
        )

    assert trace[-1]["stop_reason"] == "no_progress"
    assert any(t.get("duplicate") for t in trace)


def test_gap_driven_stop_when_satisfied():
    """基準3: missing_points 空かつ need_more=false で即停止すること。"""
    scripted = _ScriptedLLM([{"need_more": False, "missing_points": []}])
    initial = [
        s["text"] for s in _spans() if s["category_label"] == "看護記録"
    ]
    with _patch(scripted):
        collected, trace = _supplemental_search(
            _SECTION, _ENTRY, list(initial), _spans()
        )

    # 追加検索せず停止。収集内容は不変
    assert collected == initial
    assert trace[-1]["stop_reason"] == "needs_satisfied"
    # 実行された検索は無い
    assert not [t for t in trace if t.get("query")]


def test_satisfied_stop_requires_explicit_gap_analysis():
    """基準3(抜け道封鎖): missing_points 省略時は充足停止を認めないこと。

    gpt-oss が gap 分析を省き need_more=false だけ返しても、gap 分析の明示
    （missing_points キー）が無い限り needs_satisfied 停止させない。決定論
    ガードレール（no_progress）で停止し、停止理由は needs_satisfied にならない。
    """
    scripted = _ScriptedLLM(
        [
            {"need_more": False},  # missing_points キー無し（gap分析省略）
            {"need_more": False},  # 同上 → no_progress で停止
        ]
    )
    with _patch(scripted):
        _collected, trace = _supplemental_search(
            _SECTION, _ENTRY, [], _spans()
        )

    # gap 分析無しの need_more=false では充足停止しない
    assert trace[-1]["stop_reason"] != "needs_satisfied"
    assert trace[-1]["stop_reason"] == "no_progress"


def test_guardrail_caps_iterations():
    """基準4: need_more=true 連発でも MAX_SEARCH_STEPS で必ず停止すること。"""
    # 毎手 need_more=true で別カテゴリを取得し続ける（決して充足しない）
    forever = {
        "need_more": True,
        "missing_points": ["まだ足りない"],
        "tool": "category",
        "category": "看護記録",
    }
    scripted = _ScriptedLLM([dict(forever) for _ in range(20)])
    # 反復上限を 3 に固定して停止を厳密に検証
    with (
        patch("graph.nodes.section_worker.MAX_SEARCH_STEPS", 3),
        _patch(scripted),
    ):
        _collected, trace = _supplemental_search(
            _SECTION, _ENTRY, [], _spans()
        )

    # 上限で停止し、暴走していない
    assert trace[-1]["stop_reason"] in {"max_steps", "no_progress"}
    # LLM 呼び出しは上限の 3 回を超えない
    assert len(scripted.prompts) <= 3


def test_runtime_plan_pulls_routing_external_category():
    """基準5: LLM が観測から routing 外カテゴリを動的指定して拾えること。

    routing は nursing_note のみだが、LLM が「薬剤情報が不足」と観測し
    category ツールで薬剤・服薬を実行時取得する（カテゴリ越境）。
    """
    scripted = _ScriptedLLM(
        [
            {
                "need_more": True,
                "missing_points": ["内服薬の情報"],
                "tool": "category",
                "category": "薬剤・服薬",  # routing(nursing_note) 外
            },
            {"need_more": False, "missing_points": []},
        ]
    )
    initial = [
        s["text"] for s in _spans() if s["category_label"] == "看護記録"
    ]
    with _patch(scripted):
        collected, _trace = _supplemental_search(
            _SECTION, _ENTRY, list(initial), _spans()
        )

    # routing 外カテゴリの薬剤スパンが実行時に追加されている
    assert any("アムロジピン" in c for c in collected)


def test_trace_records_steps_and_stop_reason():
    """基準6: search_trace に観測・クエリ・停止理由・delta が残ること。"""
    scripted = _ScriptedLLM(
        [
            {
                "need_more": True,
                "missing_points": ["酸素"],
                "tool": "keyword",
                "keyword": "酸素",
                "reason": "酸素投与の詳細を取得",
            },
            {"need_more": False, "missing_points": []},
        ]
    )
    with _patch(scripted):
        _collected, trace = _supplemental_search(
            _SECTION, _ENTRY, [], _spans()
        )

    # 実行された検索ステップが記録されている
    executed = [t for t in trace if t.get("query")]
    assert executed and executed[0]["tool"] == "keyword"
    assert executed[0]["added_count"] >= 1
    assert executed[0]["reason"] == "酸素投与の詳細を取得"
    # 最終エントリに停止理由と coverage delta
    assert trace[-1]["final"] is True
    assert trace[-1]["stop_reason"] == "needs_satisfied"
    assert trace[-1]["coverage_delta"] >= 1


def test_graceful_fallback_on_json_failure():
    """基準6/R10: chat_json が None なら決定論収集を保ったまま停止すること。"""
    scripted = _ScriptedLLM([None])
    initial = [
        s["text"] for s in _spans() if s["category_label"] == "看護記録"
    ]
    with _patch(scripted):
        collected, trace = _supplemental_search(
            _SECTION, _ENTRY, list(initial), _spans()
        )

    # 決定論収集は失われない（網羅性の下限保証）
    assert collected == initial
    assert trace[-1]["stop_reason"] == "json_fail"


def test_coverage_report_includes_excerpt():
    """観測の質: カバレッジに代表抜粋（内容行）が含まれること。"""
    spans = _spans()
    text_to_span = {s["text"]: s for s in spans}
    collected = [s["text"] for s in spans if s["category_label"] == "看護記録"]
    report = _coverage_report(collected, text_to_span)
    assert "看護記録" in report
    assert "件" in report
    # 件数（量）だけでなく内容（質）の代表抜粋が付く
    assert "例:" in report
    assert "息苦しい" in report or "転倒" in report


def test_first_excerpt_skips_headings():
    """_first_excerpt が日付・カテゴリ見出しを飛ばし内容行を返すこと。"""
    text = "- 20230209\n  - 看護記録\n    S: 息苦しい"
    assert _first_excerpt(text) == "S: 息苦しい"


def test_synthetic_skips_supplement():
    """synthetic セクションは補完不要でスキップ（トレース空）。"""
    entry = {"mode": "synthetic", "categories": ["nursing_note"]}
    initial = ["全チャンク"]
    # chat_json は呼ばれないはず
    with patch("graph.nodes.section_worker.chat_json") as mock:
        collected, trace = _supplemental_search(
            _SECTION, entry, list(initial), _spans()
        )
    assert collected == initial
    assert trace == []
    mock.assert_not_called()
