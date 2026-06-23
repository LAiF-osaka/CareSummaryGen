"""section_worker ノード（v2・ハイブリッド検索）。

Send で起動される単一ノード。1セクション分を次の順で処理し、親
GlobalState.section_results に {section_key: SectionResult} を返す
（reducer がマージ）。

  1. collect()            … 決定論的全件収集（安全網・LLM不使用）
  2. supplemental_search  … 観測駆動 agentic 補完検索（manual ReAct ループ）
  3. _extract()           … 本文生成（空なら最大 MAX_REFILL 回再試行）
  4. absent_categories()  … 決定論的網羅点検（LLMの停止判断に依存しない）

②は LLM が観測（カバレッジ・検索履歴）→ 不足同定 → クエリ動的生成 → 言い換えを
反復し、collect() が routing 固定で取りこぼす同義語・含意・カテゴリ越境を埋める。
grep 実行・状態更新・停止判定は決定論で、停止の最終決定権は決定論側（LLM の
need_more は助言）。反復上限・進捗ゼロ検出・収集の先行・点検の後行という決定論
ガードレールで囲む（verifier-in-the-loop）。詳細: docs/agentic-search-redesign.md。
"""

from config.settings import LARGE_NUM_CTX, MAX_REFILL, MAX_SEARCH_STEPS
from graph.search_index import (
    absent_categories,
    collect,
    execute_search_tool,
)
from graph.state import SectionResult
from llm.client import chat_json
from llm.prompts import SECTION_EXTRACT_PROMPT, SUPPLEMENT_PROMPT

_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "body": {"type": "string"},
        "cited_dates": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["body"],
}

# 浅いフラット schema（gpt-oss の構造化出力安定化）。satisfied/missing で
# gap 分析（充足判定）を外在化し、tool/引数で次の1手を指示させる。
_SUPPLEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "satisfied_points": {"type": "array", "items": {"type": "string"}},
        "missing_points": {"type": "array", "items": {"type": "string"}},
        "need_more": {"type": "boolean"},
        "tool": {"type": "string"},
        "keyword": {"type": "string"},
        "category": {"type": "string"},
        "start_date": {"type": "string"},
        "end_date": {"type": "string"},
        "reason": {"type": "string"},
    },
    # missing_points を必須化し gap 分析の明示を強制する（充足停止の抜け道封鎖）。
    # ローカル Ollama は grammar で強制、Cloud は _supplemental_search 側の
    # キー存在ゲートで担保する（二重防御）。
    "required": ["need_more", "missing_points"],
}


def section_worker(payload: dict) -> dict:
    """1セクションをハイブリッド検索で生成する（Send で起動）。

    Args:
        payload: {section, routing_entry, grep_index, chunks, summary_header}。

    Returns:
        {"section_results": {section_key: SectionResult}}。
    """
    section = payload["section"]
    entry = payload["routing_entry"]
    grep_index = payload["grep_index"]
    chunks = payload["chunks"]
    summary_header = payload["summary_header"]

    # 1. 決定論的全件収集（安全網）
    collected, present_labels, _dates = collect(entry, grep_index, chunks)

    # 2. 観測駆動 agentic 補完検索（決定論ガードレール内・監査トレース付き）
    collected, search_trace = _supplemental_search(
        section, entry, collected, grep_index
    )

    # synthetic は全チャンク供給で num_ctx を拡張する
    override = (
        {"num_ctx": LARGE_NUM_CTX}
        if entry.get("mode") == "synthetic"
        else None
    )

    # 3. 生成（本文が空なら決定論カウンタで再試行）
    body: str = ""
    cited: list[str] = []
    for _attempt in range(MAX_REFILL + 1):
        body, cited = _extract(section, collected, summary_header, override)
        if body.strip():
            break

    # 4. 決定論的網羅点検（LLM の停止判断に依存しない）
    absent = absent_categories(entry, present_labels)
    review = bool(absent) or not body.strip()

    key = section["key"]
    return {
        "section_results": {
            key: SectionResult(
                section_key=key,
                body=body.strip() or "記録なし（要確認）",
                cited_dates=cited,
                missing=absent,
                review_flag=review,
                search_trace=search_trace,
            )
        }
    }


def _supplemental_search(
    section: dict,
    entry: dict,
    collected: list[str],
    grep_index: list[dict],
) -> tuple[list[str], list[dict]]:
    """観測駆動の agentic 補完検索ループ（manual ReAct）。

    決定論収集 collect() は routing 固定の categories/keywords しか集めないため、
    同義語・略語・言い換え・含意・カテゴリ越境で取りこぼしが生じる。本ループは
    LLM が「収集状況を観測 → 不足を同定 → 不足を埋める検索を1手指示 → 結果を観測
    → 言い換え」を反復してこの穴を実行時推論で埋める。grep 実行・状態更新・停止
    判定はコード側（決定論）で、LLM の役割は観測を読み次クエリ JSON を出すことに
    限定する（不安定要素を JSON 1個のパースに局所化）。

    本物の agentic search の受入基準（docs/agentic-search-redesign.md）:
        観測閉路（coverage/history を毎回注入）・gap 駆動停止（missing_points）・
        観測駆動の再定式化（tried_queries・ゼロ件で即停止しない）・決定論ガード
        レール（停止の最終決定権は決定論側・LLM の need_more は助言）・トレース。

    停止条件（決定論 OR）: missing 空かつ need_more=false / 反復上限 / 進捗ゼロ
    連続2回 / chat_json None（フォールバック）。synthetic は全チャンク供給で補完
    不要のためスキップする。

    Args:
        section: テンプレートのセクション定義。
        entry: routing のセクションエントリ。
        collected: 決定論収集済みの evidence テキスト。
        grep_index: スパン索引（{date, category_label, text}）。

    Returns:
        (補完を加えた evidence テキストのリスト, 監査トレース).
        トレースは各ステップの観測・クエリ・追加件数・停止理由を含む。
    """
    # synthetic は全チャンク供給で最も網羅的なため補完しない。
    if entry.get("mode") == "synthetic":
        return collected, []

    available_categories = sorted(
        {s["category_label"] for s in grep_index if s["category_label"]}
    )
    available_dates = sorted({s["date"] for s in grep_index if s.get("date")})
    # 観測のカバレッジ集計用に text→span を引けるようにする。
    text_to_span = {s["text"]: s for s in grep_index}

    seen = set(collected)
    tried: set[str] = set()  # 既出クエリ（reformulation の重複排除）
    trace: list[dict] = []
    initial_count = len(collected)
    no_progress = 0
    stop_reason = "max_steps"  # ループを抜けず上限到達した場合の既定

    for step in range(MAX_SEARCH_STEPS):
        # --- 観測（ReAct の Observation。決定論集計をプロンプトへ再投入） ---
        prompt = SUPPLEMENT_PROMPT.format(
            section_name=section["name"],
            section_description=section.get("description", ""),
            coverage=_coverage_report(collected, text_to_span),
            history=_format_history(trace),
            tried_queries="、".join(sorted(tried)) or "なし",
            available_categories="、".join(available_categories) or "なし",
            available_dates="、".join(available_dates) or "なし",
        )
        decision = chat_json(prompt, _SUPPLEMENT_SCHEMA)

        # --- 停止判定（最終決定権は決定論側） ---
        if decision is None:
            # 構造化出力失敗 → 決定論モードへフォールバック。
            stop_reason = "json_fail"
            break
        # gap 分析の明示（missing_points キーの存在）を充足停止の前提にする。
        # gpt-oss が gap 分析を省いて need_more=false だけ返した場合は充足停止
        # を認めず、決定論ガードレール（no_progress）で停止させる（抜け道封鎖）。
        gap_analyzed = "missing_points" in decision
        missing = [str(m) for m in decision.get("missing_points", []) if m]
        if gap_analyzed and not decision.get("need_more") and not missing:
            # gap 駆動の充足判定（gap 分析済み・missing 空・LLM が十分と判断）。
            satisfied = [
                str(s) for s in decision.get("satisfied_points", []) if s
            ]
            trace.append(
                {
                    "step": step,
                    "stop_reason": "needs_satisfied",
                    "satisfied": satisfied,
                }
            )
            stop_reason = "needs_satisfied"
            break

        tool = str(decision.get("tool", "")).strip()
        query = _query_value(tool, decision)
        if not tool or not query:
            # 不足ありだがクエリ未指定。これ以上進めない（進捗ゼロ扱い）。
            no_progress += 1
            trace.append(
                {
                    "step": step,
                    "missing": missing,
                    "added_count": 0,
                    "note": "no_query",
                }
            )
            if no_progress >= 2:
                stop_reason = "no_progress"
                break
            continue

        qkey = f"{tool}:{query}"
        if qkey in tried:
            # 繰り返し検出 → 即停止せず言い換えを1手促す（reformulation）。
            no_progress += 1
            trace.append(
                {
                    "step": step,
                    "tool": tool,
                    "query": query,
                    "added_count": 0,
                    "duplicate": True,
                }
            )
            if no_progress >= 2:
                stop_reason = "no_progress"
                break
            continue
        tried.add(qkey)

        # --- アクション（grep 実行は決定論） ---
        new_spans = execute_search_tool(tool, decision, grep_index)
        added = [s for s in new_spans if s not in seen]
        trace.append(
            {
                "step": step,
                "tool": tool,
                "query": query,
                "hit_count": len(new_spans),
                "added_count": len(added),
                "missing": missing,
                "reason": str(decision.get("reason", "")),
            }
        )
        if not added:
            # ゼロ進捗 → 即停止せず言い換えの機会を与える（R4）。連続2回で停止。
            no_progress += 1
            if no_progress >= 2:
                stop_reason = "no_progress"
                break
            continue
        no_progress = 0
        collected = collected + added
        seen.update(added)

    # 監査用に最終サマリ（停止理由・coverage delta）を残す。
    trace.append(
        {
            "final": True,
            "stop_reason": stop_reason,
            "coverage_delta": len(collected) - initial_count,
        }
    )
    return collected, trace


def _coverage_report(
    collected: list[str], text_to_span: dict[str, dict]
) -> str:
    """収集済み evidence のカバレッジを観測材料として整形する。

    collected テキストを grep_index のスパンに引き戻し、category_label 別の
    件数と代表抜粋、収集済み日付範囲を箇条書きで返す（決定論集計）。代表抜粋を
    付すことで LLM が「どこが薄いか（量）」に加え「何が書かれているか（質）」を
    観測でき、的確な不足同定・越境クエリ生成につなげる。

    Args:
        collected: 収集済み evidence テキスト。
        text_to_span: text→span の逆引き（カテゴリ・日付の復元用）。

    Returns:
        カテゴリ別件数・代表抜粋・日付範囲の箇条書き文字列。
    """
    if not collected:
        return "（まだ何も収集していない）"
    # カテゴリ別の件数と代表抜粋（最初に出会ったスパンの内容行）を集計する。
    by_category: dict[str, dict] = {}
    dates: set[str] = set()
    for text in collected:
        span = text_to_span.get(text)
        if span and span.get("category_label"):
            label = span["category_label"]
        else:
            # 補完取得スパンや label=None は別枠で件数を示す。
            label = "（カテゴリ未分類/補完取得）"
        info = by_category.setdefault(label, {"count": 0, "excerpt": ""})
        info["count"] += 1
        if not info["excerpt"]:
            info["excerpt"] = _first_excerpt(text)
        if span and span.get("date"):
            dates.add(span["date"])
    lines = []
    for cat, info in sorted(by_category.items()):
        excerpt = f"（例: {info['excerpt']}）" if info["excerpt"] else ""
        lines.append(f"- {cat}: {info['count']}件{excerpt}")
    if dates:
        lines.append(
            f"- 収集済み日付: {min(dates)}〜{max(dates)}（{len(dates)}日分）"
        )
    return "\n".join(lines)


def _first_excerpt(text: str, limit: int = 40) -> str:
    """スパン本文から代表的な内容行を1行抜粋する（観測の質向上用）。

    日付見出し（`- YYYYMMDD`）・カテゴリ見出し（`- ラベル`）行は飛ばし、最初の
    内容行を limit 文字で切って返す。内容行が無ければ見出しを除いた先頭を返す。
    """
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("- "):
            continue  # 空行・見出し行（日付/カテゴリ）はスキップ
        return stripped[:limit]
    return text.strip().lstrip("-").strip()[:limit]


def _format_history(trace: list[dict]) -> str:
    """直近3手の実行済み検索結果を観測材料として整形する。"""
    executed = [t for t in trace if t.get("query") and "added_count" in t]
    if not executed:
        return "（まだ検索していない）"
    lines = []
    for t in executed[-3:]:
        if t.get("hit_count", t.get("added_count", 0)) == 0:
            result = "ゼロ件"
        else:
            result = f"{t['added_count']}件追加"
        lines.append(f"- {t['tool']}「{t['query']}」→ {result}")
    return "\n".join(lines)


def _query_value(tool: str, decision: dict) -> str:
    """ツール別のクエリ代表値を取り出す（重複検出・履歴表示用）。"""
    if tool == "keyword":
        return str(decision.get("keyword", "")).strip()
    if tool == "category":
        return str(decision.get("category", "")).strip()
    if tool == "date_range":
        start = str(decision.get("start_date", "")).strip()
        end = str(decision.get("end_date", "")).strip()
        return f"{start}-{end}" if start and end else ""
    return ""


def _extract(
    section: dict,
    collected: list[str],
    summary_header: str,
    override: dict | None,
) -> tuple[str, list[str]]:
    """収集 evidence からセクション本文を生成する。"""
    evidence = "\n\n---\n\n".join(collected) if collected else "記録なし"
    prompt = SECTION_EXTRACT_PROMPT.format(
        section_name=section["name"],
        section_description=section.get("description", ""),
        summary_header=summary_header or "記録なし",
        evidence=evidence,
    )
    parsed = chat_json(prompt, _EXTRACT_SCHEMA, options_override=override)
    if not parsed:
        return "", []
    body = str(parsed.get("body", "")).strip()
    cited = [str(d) for d in parsed.get("cited_dates", []) if d]
    return body, cited
