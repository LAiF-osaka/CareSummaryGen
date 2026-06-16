# Agentic Search ゼロベース再設計

本書は、看護サマリ生成の検索・抽出を**ゼロベースで再設計**したものである。
現行の `plan / search / extract / evaluate / reflect / revise / synthesize` ループは踏襲しない。

設計は、具体実装（Claude Code / Morph / Chroma の agentic search）と検索系論文
（Search-R1, Self-RAG, Corrective RAG, Adaptive-RAG, APEX-Searcher）、gpt-oss の構造化出力信頼性、
臨床文書のセクション抽出に関する4観点のWeb調査、および3観点（gpt-oss信頼性/完全性・効率/LangGraph統合）の
設計レビュー（Critical 指摘を反映）に基づく。比喩・平易化・抽象化は用いない。

本設計は `GRAPH_VERSION=v2` フラグ下の新経路として実装し、現行（v1）は当面残置する（後方互換）。

---

## 1. 設計判断: agentic search は本ユースケースに最適か

### 結論

**入力規模に応じた2経路ルーティングを採用する（Adaptive-RAG の思想）。**

| 入力規模（総トークン） | 経路 | LLM呼出 |
|---|---|---|
| **≤ 約32,000 トークン**（短〜中期入院。多くの実入力） | **single-pass long-context**: 全セクションを1回の抽出で生成 → 決定論 verify → 欠損セクションのみ再生成 | 1 +（欠損再生成のみ） |
| **> 約32,000 トークン**（長期入院） | **section-routed map**: セクション単位で routing 収集 → 抽出 → verify → refill（最大1） | セクション数 +（refill のみ） |

全体を反復検索ループにする現行 v1 は、本ユースケースに構造的にミスマッチ。理由:

- 入力は Web/巨大コーパスではなく**1患者の日付チャンク化済み Markdown**（閉じた数十〜数百チャンク）。grep（部分文字列）が dense retrieval を上回る領域（調査: inline 配信で grep 93.1% vs vector 83.6%）。ベクトル DB 不要。
- **何を取りに行くかが事前に確定している**（テンプレートのセクション＋カテゴリ）。agentic にゼロから検索計画を立てる必要がない。MORPHEUS 方式（section-specific extraction + routing + 決定論的組立）が適合。
- agentic 反復が優位なのは複雑度レベル3（多基準統合・矛盾解決）のみ（調査: レベル3で +9.4pt）。看護サマリの大半は事実抽出で、全セクションに反復ループを掛けると `8 × セクション数` で実用時間を超過（調査 pitfall）。
- long-context 一括は **OSS モデルが 32k 超で劣化**（lost-in-the-middle, arXiv:2411.03538）。32k 以内なら 1 呼び出しで全セクション生成でき最安（レビュー反映: 短期入院で N→1）。32k 超でのみ section-routed map に切替。

### レビュー反映（経路選択の修正）

- **初期レビュー設計は「常に N 並列」だったが、レビュー2 の指摘により ≤32k は single-pass を既定**に格上げした。短期入院で常に N 並列を回すのは非効率（単一 Ollama では N×1パスに直列化されレイテンシも改善しない、後述 §7）。
- single-pass で欠損・未充足セクションが出た場合のみ、当該セクションを section-routed map で個別再生成する。

---

## 2. グラフ構造（LangGraph 新規定義）

```
                 ┌──────────┐
        START ──▶│  ingest  │ 入力正規化・日付チャンク化・grep索引・header抽出・routing解決・規模判定
                 └────┬─────┘
                      │  route_by_size（決定論: 総トークン ≤ TH か）
            ┌─────────┴──────────┐
            ▼ (≤TH)              ▼ (>TH)
      ┌───────────┐       ┌────────────────────────────┐
      │ single_   │       │  fanout_sections (Send×N)   │
      │ pass      │       │  各セクション = 単一ノード    │
      │ (全節1回) │       │  section_worker（内部bounded │
      └─────┬─────┘       │   loop: collect→extract→    │
            │ verify欠落   │   verify→refill 最大1）      │
            │ セクションのみ└───────────┬────────────────┘
            │ Send で個別  　            │ reducer で section_results 集約
            └────────┬───────────────────┘
                     ▼
               ┌───────────┐
               │ assemble  │ section_delimiter で決定論組立・欠損は「記録なし（要確認）」明示
               └─────┬─────┘
                     ▼
               ┌───────────┐
               │consistency│ atomic-claim を grep_index で照合・未支持を review_flag（自動修復しない）
               └─────┬─────┘
                     ▼
               ┌───────────┐
               │ finalize  │ final_summary 確定・review_flags 付与
               └─────┬─────┘
                     ▼
                    END
```

### ノード一覧

| ノード | 責務 | LLM |
|---|---|---|
| `ingest` | 入力受領、`- YYYYMMDD` 日付チャンク化、`grep_index` 構築、`summary_header` 抽出、routing 解決（enum→ラベル）、総トークン判定 | なし |
| `single_pass` | 全日付チャンク（+header）を1回供給し全セクションを抽出生成。`format` schema。欠損セクションを返す | あり（1回） |
| `section_worker`（`Send` 並列・**単一ノード関数**） | 1セクションの collect→extract→verify→refill を関数内ループ（最大2パス）で完結し `{section_key: SectionResult}` を返す | あり |
| `assemble` | `section_delimiter`（`--- {name} ---`）で決定論組立。空/未充足は「記録なし（要確認）」明示 | なし |
| `consistency` | draft の atomic claim を grep_index で照合。未支持 claim を review_flag 化（**自動修復せず人手レビュー**） | claim分解のみ |
| `finalize` | final_summary 確定、review_flags 付与 | なし |

> **レビュー反映（R3 #3）**: `section_worker` は**サブグラフではなく単一ノード関数**として実装する。`Send` で起動された単一ノードが親 `GlobalState.section_results`（reducer 付き channel）に `{sk: SectionResult}` を返すことで LangGraph が確実にマージする。サブグラフのローカル channel は親へ自動伝播しないため、サブグラフ化は採らない。内部の collect/extract/verify/refill は関数内の `for _ in range(2)` ループで回す。

### 条件分岐（エッジ）

- `ingest → route_by_size`: 総トークン `≤ TH`（既定 32768）なら `single_pass`、超なら `fanout_sections`。
- `single_pass → 個別 Send | assemble`: 決定論 verify で欠損セクションがあれば当該セクションのみ `Send("section_worker", ...)`、無ければ `assemble`。
- `section_worker → （reducer 集約）→ assemble`: 全 worker 完了で `assemble`（並列なので「次へ」概念なし）。
- `consistency → finalize`: 未支持 claim はフラグ化のみ。**自動再生成しない**（医療の偽陽性リスク。調査 pitfall）。

---

## 3. 検索/抽出の具体メカニズム

### 3.1 チャンク粒度と grep_index

- **基本粒度 = 日付チャンク（`- YYYYMMDD` 単位）**。既存 `input_adapter._build_search_index` の不変条件をそのまま採用。
- `ingest` の `_explode_to_spans` が各チャンクを **(date, category_label, span_text)** のレコード集合 `grep_index` に展開する。
- DB renderer 経路では `CATEGORY_LABELS` の小見出し（`  - 看護記録` 等）が category アンカーになる。

> **レビュー反映（R3 #2: 非DB入力対策）**: カテゴリ小見出しを出すのは DB renderer 経路（`build_context`）のみ。テキスト/XML 由来の Markdown（`data/test_sample1.md` 等、現行主入力）は `- YYYYMMDD` 境界はあるが `  - {label}` 小見出しを持たない。`_explode_to_spans` は小見出しが無いチャンクを **`category_label=None` の単一 span** として扱い、第1層（カテゴリ悉皆）ではなく第2層（keyword grep）と synthetic 経路で拾う。第1層の網羅保証は DB 経路限定であることを明示する。

### 3.2 検索の3層（ベクトル不使用）

1. **構造検索（カテゴリ悉皆・決定論・最優先）**: routing が指定する category（enum 値）を `CATEGORY_LABELS` でラベルへ解決し、`grep_index` の該当ラベル span を**全件回収**（top-k で切らない）。これが網羅性の主担保。
2. **keyword grep（補完）**: セクション description 由来のキーワード（例: 指導 → `指導|説明|教育|パンフレット`）で正規表現マッチ。カテゴリ越境・未分類（label=None）span を拾う。
3. **synthetic セクションは全日付チャンク供給**（検索しない。§6）。

> **レビュー反映（R2/R3 #1: enum↔ラベル写像）**: routing YAML には `RecordCategory` の **enum 値**を書き、`ingest` が `CATEGORY_LABELS` を介して**ラベルへ解決してから** grep する（写像の単一真実源）。enum 値で直接 grep すると `procedure`（ラベルは「処置・医療機器」）等が**全件ヒット0**になるため。§10 の契約テストで「全 routing の categories が `RecordCategory` メンバかつ `CATEGORY_LABELS` にキー存在」を必須化する。

> **レビュー反映（R2: Unlabeled span の漏れ）**: 未分類（label=None）span はどのカテゴリ悉皆にも入らないため、全 extractive セクションの keyword grep 対象プールに含める。

### 3.3 完全性の保証（3層）

- **第1層 入力側明示**: DB renderer が `missing_categories` を「記録なし: ◯◯」として出力済み、Unlabeled は header に保持済み。
- **第2層 verify（決定論・二段）**: 後述 §4。
- **第3層 assemble/finalize**: 未充足項目は空欄でなく「記録なし（要確認）」と明示し `review_flags` を立てる。

---

## 4. 終了条件（LLMスコア不使用・決定論）

ユーザ方針どおりスコア（0–1関連度）も「SUFFICIENT」LLM判定も停止に使わない。

### 4.1 verify を二段にする（レビュー反映: R1, R2, R3 #6）

初期設計は「`required_items` の文字列照合」を停止の肝としたが、レビューにより**`required_items` を空にすると verify が常に done を返し refill が一度も発火しない**（第2層がデッドコード化）ことが判明した。これを次の二段照合に置換する。空 `required_items` でも機能する。

1. **evidence 充足判定（refill の起点）**: routing の各 category が `collected` span に1件以上存在するか。存在しない category があれば → `refill`（keyword 拡張で再収集）。「入力にある情報の取りこぼし」を検出する本体。
2. **body 充足判定（再 extract の起点）**: `collected` に存在する category の情報が、抽出本文 `body` に反映されたか（当該 category の代表語 or `cited_dates` 非空で判定）。evidence にあるのに body に無ければ → 再 extract。reasoning 混入で body が汚れたケースを救済する。

`required_items` を定義する場合は追加チェックとして使うが、**空でも (1)(2) で機能する**。

### 4.2 停止条件（OR）

| 判断点 | 完了条件（OR） |
|---|---|
| section_worker 内部ループ | (a) evidence 充足 かつ body 充足、OR (b) `refill_count >= 1`（ハード上限）、OR (c) refill しても `collected` span 集合のハッシュが不変（限界効用ゼロ） |
| 全体完了 | 全 section_worker 完了。`consistency` は未支持 claim をフラグ化するのみで再生成ループを作らない |

- `verify` は **LLM を使わない純照合関数**。gpt-oss の構造化不安定性を停止経路から完全排除する。

---

## 5. 構造化出力・判断の信頼性確保（gpt-oss）

原則: **停止判断に JSON 構造を要求しない／生成にだけ format を使う／必ず堅牢パース＋フォールバック**。

### 5.1 extract（本文生成）

- **`think=False` を明示**（format と thinking は排他, Ollama #10538）。現行 `chat()` は think を渡さず gpt-oss のデフォルト thinking が混入しうる → `chat()` に `think` 引数を追加。
- `format=ExtractResult.model_json_schema()` + `temperature=0`（self-host で GBNF 制約）。

```python
class ExtractResult(BaseModel):
    reasoning: str            # CoT 吸収（停止判断には使わない）
    body: str                 # セクション本文
    cited_dates: list[str]    # 根拠日付（verify body 充足・consistency・監査）
```

- パースは **必ず `extract_json()` 経由**（既存実装を再利用）+ Pydantic 検証 + bounded retry（最大3、エラー文は最新500字のみ付加）。
- **Ollama Cloud（`ENV==test`）は format が強制されない**ため、format は「ヒント」とみなし `extract_json()`+retry に必ず通す（調査の Cloud サイレント失敗 pitfall）。

### 5.2 verify / consistency

- `verify`: §4.1 のとおり LLM・JSON 不使用の純照合。
- `consistency`: claim 分解のみ LLM（`format: {claims: list[str]}`）、各 claim の支持判定は **grep_index への部分文字列/正規表現照合（決定論）**。LLM-as-judge を停止・修復経路に入れない（医療の偽陽性リスク）。

### 5.3 共通

- tool calling は使わない（`<|call|>` EOS 未登録バグ）。format schema + プロンプト指定に統一。
- temperature=0。繰り返しトークン検出時のみ 0.1–0.2 へ。

---

## 6. 128k コンテキストの使い分け（決定論しきい値）

| 場面 | 供給方式 | しきい値 |
|---|---|---|
| **single-pass（≤TH）** | 全日付チャンク+header を1回供給し全セクション生成 | 総トークン ≤ TH（既定 32768） |
| **extractive セクション（>TH 経路）** | routing カテゴリの span のみ供給（絞る） | 常に絞る |
| **synthetic セクション（>TH 経路）** | 全日付チャンクを時系列で全供給 | 当該呼び出しのみ `num_ctx=32768` |
| **超長期（synthetic でも >TH）** | incremental refine（date 単位カバレッジ検査付き） | §3 のカバレッジ検査で脱落 date をフラグ化 |

> **レビュー反映（R2: synthetic の完全性）**: synthetic セクションにも軽量な決定論カバレッジ検査を入れる。「全日付チャンクの各 date が最終 body の `cited_dates` に1回以上現れるか」を verify し、欠落 date を `review_flag`。refine フォールバック時も date 単位で継続チェックし脱落 date をフラグ化する（refine 段の lost-in-the-middle 再生産を検出）。

> **レビュー反映（R3 #5: num_ctx 上書き）**: 現行 `chat()` は `temperature` しか上書きできない。`chat()` に `options_override: dict | None` を追加し `options.update(options_override)` する。LLM クライアント変更点は **`think` と `options_override` の2点**（初期設計の「think 1点だけ」は誤り）。

---

## 7. 効率: v1 問題の構造的解消とコスト

| v1 の問題 | v2 での解消 |
|---|---|
| evaluate→search の**グローバル再検索ループ多発** | グローバル反復を廃止。停止はカテゴリ悉皆＋決定論二段 verify |
| `max_results=5` 取りこぼし | **悉皆回収（カテゴリ全 span）**が主経路。top-k 切り出しをしない |
| reasoning trace 混入でパース失敗 | `think=False` + `extract_json()` + retry |
| `plan` の JSON 不安定 | `plan` ノード自体を廃止 |

### 7.1 コスト見積もり（レビュー反映 R2: best/expected/worst）

`N`=セクション数。`r`=refill 発生率、`t`=retry 発生率。

| ケース | LLM 呼出数 |
|---|---|
| **best（≤TH, single-pass 成功）** | **1** |
| **expected（≤TH, 一部欠落）** | `1 + (欠落セクション数) × (1 + 期待retry)` |
| **worst（>TH, 全セクション refill+retry）** | `N × (1 + r) × (1 + 期待retry) + claim分解` |

`N=6`, `r=t` 最悪で extract 系 ≈ 36 呼出（上限は静的に決まる）。**多くの実入力（≤TH）は 1 呼出**で完結する点が v1 比の最大の改善。

### 7.2 レイテンシの注意（レビュー反映 R2: 単一 Ollama）

gpt-oss:120b は1リクエストで GPU を専有するため、単一 Ollama では `Send` の並列ファンアウトは**サーバ側でキュー直列化**され、レイテンシは `N×1パス` に戻る。
- 並列の効果は「コード構造の独立性」であって、単一 Ollama ではレイテンシ削減にならない。
- レイテンシ削減を狙う場合は `OLLAMA_NUM_PARALLEL` または複数バックエンドが前提（§10 検証に明記）。
- この点でも **≤TH の single-pass（1呼出）が単一 Ollama では最速**であり、§1 の経路選択を補強する。

---

## 8. LangGraph 実装スケッチ

### 8.1 State

```python
from typing import Annotated, Optional, TypedDict
from operator import add


def merge_sections(left: dict, right: dict) -> dict:
    """section_key 単位で集約する reducer（各 worker は自分の key だけ書く）。"""
    return {**left, **right}


class SectionResult(TypedDict):
    section_key: str
    body: str
    cited_dates: list[str]
    missing: list[str]          # 未充足（evidence/body）。finalize で明示
    review_flag: bool


class GlobalState(TypedDict):
    # 入力
    patient_id: str
    raw_context: str
    hospital: str
    template_id: str
    # ingest 成果物
    template: dict
    summary_header: str
    chunks: list[str]
    grep_index: list[dict]          # {date, category_label, text}
    routing: dict                   # section_key -> {mode, categories, keywords, required_items}
    total_tokens: int
    # 集約（並列）
    section_results: Annotated[dict[str, SectionResult], merge_sections]
    # 出力
    draft_summary: str
    final_summary: str
    review_flags: Annotated[list[str], add]
    global_repair_count: Annotated[int, add]   # 並列書き込み競合回避（R3 #4）
    error: Optional[str]
```

> **レビュー反映（R3 #4）**: `global_repair_count` は `Annotated[int, add]`（reducer 付き）にして並列 `Send` の同時書き込み競合（InvalidUpdateError）を回避する。`SectionState` 側の reducer 定義は不要（単一ノード関数化により削除）。

### 8.2 主要ノード（単一ノード関数の section_worker）

```python
def ingest(state: GlobalState) -> dict:
    """正規化・日付チャンク化・grep索引・routing解決・規模判定。"""
    template = load_template(state["template_id"])
    header = _extract_summary_header(state["raw_context"])      # 既存流用
    chunks, idx = _build_search_index(state["raw_context"])     # 既存流用
    grep_index = _explode_to_spans(chunks, idx)                 # label=None フォールバック含む
    routing = load_routing(state["template_id"])                # enum値で記述
    total = count_tokens(state["raw_context"])
    return {"template": template, "summary_header": header,
            "chunks": chunks, "grep_index": grep_index, "routing": routing,
            "total_tokens": total, "section_results": {},
            "global_repair_count": 0}


def route_by_size(state: GlobalState) -> str:
    """≤TH なら single_pass、超なら fanout。"""
    return "single_pass" if state["total_tokens"] <= TH else "fanout"


def section_worker(state: dict) -> dict:
    """Send で起動される単一ノード。collect→extract→verify→refill を内包。

    親 GlobalState.section_results に {section_key: SectionResult} を返す。
    """
    section = state["section"]
    entry = state["routing_entry"]
    collected = _collect(entry, state["grep_index"], state["chunks"])  # enum→label解決済
    collect_hash = _hash(collected)

    body, cited = "", []
    for _ in range(2):  # 初回 + refill 最大1
        body, cited = _extract(section, collected, state["summary_header"])
        ev_missing = _evidence_missing(entry, collected)   # §4.1 (1)
        body_missing = _body_missing(entry, collected, body, cited)  # §4.1 (2)
        if not ev_missing and not body_missing:
            break
        if ev_missing:
            expanded = _grep_expand(state["grep_index"], ev_missing)
            new = _dedup(collected + expanded)
            if _hash(new) == collect_hash:   # 限界効用ゼロ
                break
            collected, collect_hash = new, _hash(new)
        # body_missing のみなら再 extract（collected 据え置き）

    sk = section["key"]
    missing = _evidence_missing(entry, collected) + _body_missing(
        entry, collected, body, cited
    )
    return {"section_results": {sk: SectionResult(
        section_key=sk, body=body, cited_dates=cited,
        missing=missing, review_flag=bool(missing))}}


def assemble(state: GlobalState) -> dict:
    """section_delimiter で決定論組立。欠損は明示（LLM不使用）。"""
    parts = []
    for s in state["template"]["sections"]:
        r = state["section_results"].get(s["key"])
        body = r["body"] if r and r["body"] else "記録なし（要確認）"
        parts.append(state["template"]["section_delimiter"].format(name=s["name"]))
        parts.append(body)
    return {"draft_summary": "\n".join(parts)}
```

`chat()` の変更点（2点のみ）:
```python
def chat(prompt, *, system="", format_schema=None, temperature=None,
         think: bool = False, options_override: dict | None = None) -> str:
    options = {**LLM_OPTIONS}
    if temperature is not None:
        options["temperature"] = temperature
    if options_override:
        options.update(options_override)   # num_ctx 等のノード単位上書き
    ...
    response = ollama_client.chat(..., think=think)
```

---

## 9. 既存入力 / テンプレート / DB入力設計との接続

| 接続先 | 接続点 | 変更要否 |
|---|---|---|
| `input_adapter` | `_build_search_index`（`- YYYYMMDD` 分割）と `_extract_summary_header` を `ingest` が流用。`raw_context` 契約不変 | 流用 |
| DB入力設計（renderer/pipeline） | `build_context()` の `# 患者ID:` + `## サマリ基本情報` + `- YYYYMMDD` Markdown が `raw_context` に入る。`CATEGORY_LABELS` 小見出しが grep アンカー | 変更なし |
| テンプレート YAML | `sections[].{name,key,description}` と `section_delimiter` を流用 | 変更なし |
| **新規 routing YAML**（`templates/<id>.routing.yaml`） | section key → `{mode, categories(enum値), keywords, required_items}` | 新規追加 |

routing YAML 例（hanwa, 抜粋。enum 値で記述し ingest がラベル解決）:
```yaml
sections:
  instruction:
    mode: extractive
    categories: [discharge_support, nursing_note]   # → "退院支援・継続課題","看護記録"
    keywords: ["指導", "説明", "教育", "パンフレット"]
    required_items: []        # 空でも §4.1 二段 verify で機能する
  medical_equipment:
    mode: extractive
    categories: [procedure, medication]             # → "処置・医療機器","薬剤・服薬"
    keywords: ["カテーテル", "ドレーン", "挿入", "装着"]
  nursing_process:
    mode: synthetic           # 全日付チャンク供給 + date カバレッジ検査
    categories: [nursing_note, vital_sign, adl, nursing_problem]
  risks:
    mode: synthetic
    categories: [nursing_problem, risk_assessment, discharge_support]
    keywords: ["継続", "リスク", "転倒", "再発"]
```

---

## 10. 移行計画と検証

### 置換手順（後方互換）

1. **新ファイル追加（既存非破壊）**: `graph/state_v2.py`、`graph/nodes_v2/{ingest,single_pass,section_worker,assemble,consistency,finalize}.py`、`graph/builder_v2.py`、`templates/<id>.routing.yaml`、`templates_loader/routing.py`。
2. **LLM クライアント変更（2点）**: `chat()` に `think: bool=False` と `options_override: dict|None=None` を追加。`extract_json()` は再利用。
3. **routing 整備**: hanwa(6)/shinkinen(2) の routing YAML 作成。`categories` を `RecordCategory` 値・`CATEGORY_LABELS` キーと突合（契約テストで保証）。
4. **builder 切替**: `app.py` のグラフ取得を `GRAPH_VERSION=v2` でフラグ切替（旧 builder 残置）。
5. **旧ノード撤去**: v2 が E2E 安定後、`graph/nodes/{plan,search,extract,evaluate,reflect,revise,synthesize}.py`・旧 edges・旧 state を削除。

### 検証

- **契約テスト（最重要・レビュー反映 R2/R3 #1）**: 全 routing の `categories` が `RecordCategory` メンバかつ `CATEGORY_LABELS` にキー存在。`ingest` の enum→ラベル解決後に grep_index と突合してヒットすること。
- **非DB入力テスト（R3 #2）**: 小見出しの無い `data/test_sample1.md` 等で `_explode_to_spans` が label=None フォールバックし、keyword grep / synthetic でカバレッジが破綻しないこと。
- **ユニット**: `_collect` カテゴリ悉皆（取りこぼし0）、`verify` 二段（evidence/body）、内部ループ終了（充足/上限/ハッシュ不変）、`assemble` 欠損明示、`global_repair_count` の並列加算。
- **E2E（mocked LLM）**: hanwa 6 / shinkinen 2 セクション全出力、空セクションが「記録なし（要確認）」、review_flags 起動、single-pass（≤TH）と fanout（>TH）の両経路。
- **E2E（real Ollama）**: `think=False`+format で extract が JSON パース成功（reasoning 混入で失敗しない）。Cloud で format 非強制でも `extract_json()`+retry で復旧。
- **回帰**: `data/test_sample1.md` 等で v1 と v2 の出力を並置し、カテゴリ取りこぼしが v2 で減ること。
- **完了基準（DoD）**: フラグ切替で旧経路維持、Critical 0、docstring・型注釈。

---

## 11. レビューで確認された未解決リスク（実装時の前提）

| # | リスク | 対応方針 |
|---|---|---|
| 1 | grep の網羅保証は **enum→ラベル写像の正しさに依存**（誤れば全件0ヒット） | ingest で単一写像、契約テスト必須 |
| 2 | 第1層カテゴリ悉皆は **DB renderer 経路限定**。非DB入力では keyword/synthetic 依存 | label=None フォールバック、非DB入力テスト |
| 3 | `required_items` 空でも機能させるため verify を **evidence/body 二段**に再定義 | §4.1 |
| 4 | 単一 Ollama では `Send` 並列が直列化されレイテンシ改善しない | ≤TH は single-pass を既定、並列効果は構造独立性に限定 |
| 5 | synthetic の完全性は date カバレッジ検査のみ（refine で情報脱落しうる） | date 単位カバレッジ検査＋フラグ化 |
| 6 | consistency の claim 照合は偽陽性で事実誤認に直結しうる | 自動修復せずフラグ＋人手レビュー |
| 7 | コストは入力規模依存（best 1 / worst ≈36 呼出） | §7.1 の3点見積もり |
