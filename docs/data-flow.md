# データフロー仕様: 入力 → 処理 → 出力（v2）

本書は CareSummaryGen の `POST /ask`（テキスト入力）と `POST /ingest`（DB入力）における
入力データ・処理フロー・出力データを、実装上の関数名・State フィールド名・ノード名に対応づけて記述する。

構成: LangGraph + Ollama Python SDK + FastAPI / agentic search v2（入力規模ルーティング）。
詳細設計: [agentic-search-redesign.md](agentic-search-redesign.md)（処理）、[db-input-design.md](db-input-design.md)（DB入力）。

---

## 1. 入力

入力経路は2つある。どちらも最終的に同一の `_run_graph(context, patient_id, template_id)`（`app.py`）へ収束する。

### 1.1 POST /ask（テキスト入力）

- 定義: `app.py` の `AskRequest`

| フィールド | 型 | 必須 | デフォルト | 説明 |
|---|---|---|---|---|
| `context` | `str` | 必須 | なし | 医療記録テキスト（日付チャンク Markdown） |
| `patient_id` | `str` | 任意 | `"unknown"` | 患者ID |
| `template_id` | `str \| None` | 任意 | `None` | `None` のとき環境変数 `HOSPITAL`（既定 `hanwa`） |

### 1.2 POST /ingest（DB入力）

- 定義: `app.py` の `IngestRequest`

| フィールド | 型 | 必須 | デフォルト | 説明 |
|---|---|---|---|---|
| `patient_id` | `str` | 必須 | なし | 患者ID（正規表現 `[A-Za-z0-9_-]+` で検証） |
| `encounter_id` | `str` | 必須 | なし | 入院ID（同上） |
| `query_spec_id` | `str` | 任意 | `"sql_sample"` | `query_specs/<id>.yaml` |
| `template_id` | `str \| None` | 任意 | `None` | 省略時 `HOSPITAL` |

`/ingest` は `build_context_from_db(patient_id, encounter_id, query_spec_id)`（`adapters/pipeline.py`）で
DB から context を構築してから `_run_graph` に渡す（§4）。

### 1.3 context のテキスト形式（両経路共通の契約）

`ingest` ノード（`graph/nodes/ingest.py`）の `_build_search_index()` は、正規表現 `^- (\d{8})\s*$`
（行頭の `- YYYYMMDD`）をチャンク境界として分割する。最初の日付行より前の本文は `_extract_summary_header()`
が `summary_header` として抽出する（`# 患者ID:` 行は除外）。`explode_to_spans()` が各日付チャンクを
`  - {カテゴリ名}` 小見出し単位の `grep_index`（`{date, category_label, text}`）へ展開する。小見出しが
無いチャンクは `category_label=None` の単一スパンになる。

---

## 2. 処理フロー（LangGraph v2）

`_run_graph` が `GlobalState`（`graph/state.py`）初期値を構築し、`build_nursing_summary_graph()`
（`graph/builder.py`）でコンパイルしたグラフを `invoke` する。

```
START
  → ingest
  → [route_and_fanout]  ── 総トークン total_tokens で分岐
       ≤ SINGLE_PASS_TOKEN_THRESHOLD(既定32768) → single_pass
       > 閾値                                   → section_worker(Send × セクション数)
  single_pass
  → [after_single_pass]  ── 本文が空のセクションのみ section_worker へ Send、無ければ assemble
       → section_worker(Send × 欠損数)
       → assemble
  section_worker  ──(reducer merge_sections で section_results 集約)──→ assemble
  assemble → consistency → finalize → END
```

### 2.1 ノード別仕様

| ノード | ファイル | LLM 呼出 | 主な入力 | 主な出力 |
|---|---|---|---|---|
| `ingest` | `graph/nodes/ingest.py` | なし | `raw_context`, `template_id` | `template`, `routing`, `summary_header`, `chunks`, `grep_index`, `total_tokens` |
| `single_pass` | `graph/nodes/single_pass.py` | あり（1回） | `chunks`, `summary_header`, `template` | `section_results`（全セクション） |
| `section_worker` | `graph/nodes/section_worker.py` | あり | Send ペイロード（section, routing_entry, grep_index, chunks, summary_header） | `section_results`（1セクション） |
| `assemble` | `graph/nodes/assemble.py` | なし | `section_results`, `template` | `draft_summary` |
| `consistency` | `graph/nodes/consistency.py` | あり（claim分解のみ） | `draft_summary`, `grep_index` | `review_flags`（未支持claim） |
| `finalize` | `graph/nodes/finalize.py` | なし | `draft_summary`, `section_results` | `final_summary`, `review_flags` |

### 2.2 各ノードの処理内容

#### ingest
- `load_template(template_id)` / `load_routing(template_id)` をロード（routing の `categories` は `RecordCategory` の enum 値、`CATEGORY_LABELS` でラベル解決可能であることが検証済み）。
- `_extract_summary_header()` で患者横断情報を `summary_header` に抽出（検索ヒットに依存せず常時供給）。
- `_build_search_index()` で日付チャンク `chunks`/`chunk_index` を生成。
- `explode_to_spans()` で `grep_index`（`{date, category_label, text}`）を構築。
- `tiktoken`（cl100k_base）で `total_tokens` を算出。

#### route_and_fanout（条件付きエッジ・`graph/edges/routing_v2.py`）
- `total_tokens <= SINGLE_PASS_TOKEN_THRESHOLD` → `"single_pass"`。
- それ以外 → 全セクションを `Send("section_worker", payload)` で並列起動。

#### single_pass
- 全セクション仕様 + `summary_header` + 全 `chunks` を1プロンプトに入れ、`chat_json`（`format` + `extract_json` + bounded retry、`think=False`、`num_ctx=LARGE_NUM_CTX`）で `{sections: [{section_key, body, cited_dates}]}` を取得。
- 各テンプレートセクションに `SectionResult` を生成。本文が空のセクションは `review_flag=True`。

#### after_single_pass（条件付きエッジ）
- 本文が空のセクションがあれば、それらだけ `Send("section_worker", ...)`。無ければ `"assemble"`。

#### section_worker（単一ノード・Send で起動）
- `collect(entry, grep_index, chunks)`（`graph/search_index.py`）: synthetic は全日付チャンク供給、extractive はカテゴリ悉皆（top-k 制限なし）＋ keyword grep（label=None スパンも対象）。
- `_extract()`: `SECTION_EXTRACT_PROMPT` + `chat_json`（`{reasoning, body, cited_dates}`）。本文が空なら最大 `MAX_REFILL+1` 回まで再試行（決定論カウンタ。LLM スコア不使用）。
- `absent_categories(entry, present_labels)`: routing が要求するが記録に存在しないカテゴリラベルを欠落として返す。欠落があれば `review_flag`。
- `{section_key: SectionResult}` を返し、reducer `merge_sections` が `section_results` に集約。

#### assemble
- `section_delimiter`（既定 `--- {name} ---`）でテンプレート順に決定論組立。空・未生成セクションは `"記録なし（要確認）"`。LLM 不使用。

#### consistency
- `CONSISTENCY_PROMPT` + `chat_json` で `{claims: [...]}` を分解。
- 各 claim の主要トークン（数値・語）が `grep_index`＋`summary_header` に過半含まれるかを決定論照合。未支持 claim を `review_flags` に追加（自動修復しない）。

#### finalize
- `final_summary = draft_summary`。`review_flag` が立った `section_key` を `review_flags` に集約。

### 2.3 LLM 呼び出し（`llm/client.py`）
- `chat(prompt, format_schema, temperature, think=False, options_override)`: `think=False` で reasoning 混入を防ぎ、`options_override` で `num_ctx` をノード単位上書き。
- `chat_json(prompt, schema, max_retries=3)`: `chat` + `extract_json`（コードフェンス除去・平衡括弧抽出）+ retry。Ollama Cloud で `format` 非強制でも復旧。
- モデル/接続先は `ENV` で切替（production=ローカル `gpt-oss:120b`、test=`gpt-oss:120b-cloud`）。

### 2.4 ループ上限（決定論カウンタ）

| ループ | 制御 | 上限 |
|---|---|---|
| section_worker 内部の再抽出 | `MAX_REFILL`（既定1） | 本文が空のときのみ再試行 |
| single_pass の欠損補完 | after_single_pass | 欠損セクションを1回だけ section_worker へ |

LLM スコア（0–1）や「SUFFICIENT」LLM 判定は停止条件に**使わない**。

---

## 3. 出力

- 定義: `app.py` の `AskResponse`（`/ask`・`/ingest` 共通）

| フィールド | 型 | 説明 |
|---|---|---|
| `answer` | `str` | `final_summary`。テンプレート区切りの看護サマリー |
| `template_id` | `str` | 使用テンプレートID |
| `review_flags` | `list[str]` | 人手レビュー対象（欠落セクションキー・未支持 claim） |

`answer` はテンプレート（`templates/<id>.yaml`）の `section_delimiter` で区切られる。
`hanwa` は6セクション、`shinkinen` は2セクション。

---

## 4. DB入力パイプライン（/ingest 経路）

`build_context_from_db(patient_id, encounter_id, spec_id)`（`adapters/pipeline.py`）:

```
load_query_spec(spec_id)             # query_specs_loader: Pydantic検証 + バインド変数検証
  → get_adapter(source_type).fetch() # adapters/base 登録の RecordSourceAdapter（sql 等）
  → normalize(raw, spec, ...)        # adapters/normalizer: role別マッピング + コード解決 + cross_cutting
  → sample(record_set, spec)         # adapters/sampler: extremes(first/last/min/max)
  → mask(record_set)                 # adapters/phi_masker: 自由記述の電話/郵便/メール等を非可逆マスク
  → render(record_set)               # adapters/markdown_renderer: # 患者ID + サマリヘッダ + - YYYYMMDD
  → context（= AskRequest.context と同形式）
```

- `query_spec` は「論理層（record_category / columns(role) / contributes_to / sampling）」と「retrieval層（source_type 固有: `sql`）」に分離。
- `retrieval.sql` のバインド変数は `:patient_id` / `:encounter_id` のみ許可（ローダーで検証、SQLインジェクション防止）。
- `role=item` かつ `codesystem` 指定の列はコード→名称解決（例: `31001368` → `体温`）。未解決は `[未解決コード:...]`。
- 患者横断カテゴリ（`allergy`, `infection`, `nursing_problem`, `patient_profile`, `nursing_acuity`）は `cross_cutting=True` でサマリヘッダ領域へ。

---

## 5. GlobalState フィールド一覧（`graph/state.py`）

| フィールド | 型 | 用途 |
|---|---|---|
| `patient_id` / `raw_context` / `hospital` / `template_id` | `str` | 入力 |
| `template` / `routing` | `dict` | ingest がロード |
| `summary_header` | `str` | 患者横断情報（常時供給） |
| `chunks` | `list[str]` | 日付チャンク本文 |
| `grep_index` | `list[dict]` | `{date, category_label, text}` スパン |
| `total_tokens` | `int` | 入力規模（経路選択） |
| `section_results` | `Annotated[dict[str, SectionResult], merge_sections]` | セクション別結果（並列集約） |
| `draft_summary` / `final_summary` | `str` | 組立・最終出力 |
| `review_flags` | `Annotated[list[str], add]` | レビュー対象（並列加算） |
| `error` | `Optional[str]` | エラー |

`SectionResult`: `{section_key, body, cited_dates, missing, review_flag}`。
