# データフロー仕様: 入力 → 処理 → 出力（v2）

本書は CareSummaryGen の入力データ・処理フロー・出力データを、実装上の関数名・State フィールド名・ノード名に対応づけて記述する。

構成: LangGraph + Ollama Python SDK + FastAPI / agentic search v2。
詳細設計: [agentic-search-redesign.md](agentic-search-redesign.md)（処理）、[db-input-design.md](db-input-design.md)（DB入力）。
図解版: [data-flow-report.html](data-flow-report.html)。

---

## 1. 入力

入力経路は2つある。**本番の主経路はデータベースからの取得（`POST /ingest`）**であり、テキスト入力（`POST /ask`）は同形式の `context` を直接渡す代替経路である。どちらも `_run_graph(context, patient_id, template_id)`（`app.py`）に収束する。

### 1.1 主経路: POST /ingest（DBから各項目を取得）

`IngestRequest`（`app.py`）で患者ID・入院ID・取得仕様IDを受け取り、`build_context_from_db()`（`adapters/pipeline.py`）が **`query_spec` で定義された各テーブル・各カラム（項目）を SQL で取得**して `context` を構築する。

| フィールド | 型 | 必須 | デフォルト | 説明 |
|---|---|---|---|---|
| `patient_id` | `str` | 必須 | なし | 患者ID（正規表現 `[A-Za-z0-9_-]+` で検証） |
| `encounter_id` | `str` | 必須 | なし | 入院ID（同上） |
| `query_spec_id` | `str` | 任意 | `"sql_sample"` | 取得仕様（`query_specs/<id>.yaml`） |
| `template_id` | `str \| None` | 任意 | `None` | 省略時 `HOSPITAL`（既定 `hanwa`） |

#### 取得される各項目（`query_specs/sql_sample.yaml` の例）

`query_spec` は「どのテーブルのどのカラムを、どの意味役割（`role`）で取得するか」を定義する。各カラムが `role` を通じて `ClinicalRecord` のフィールドへ写像される。**カラム名が未確定でも、`query_spec` の追加・修正のみで取得項目を変えられる（コード変更不要）。**

| DBテーブル | 取得カラム | role | → ClinicalRecord フィールド | 寄与セクション |
|---|---|---|---|---|
| `vital_signs` | `measured_at` | datetime | レコードの日付（`event_date`） | — |
| | `item_code` | item(codesystem=medis_obs) | フィールド名（コード→名称解決。例 `31001368`→`体温`） | nursing_process, medical_equipment |
| | `value` | value | フィールド値 | |
| | `unit` | unit | フィールド単位 | |
| | `recorder_name` | recorder | 記録者 | |
| `nursing_records` | `recorded_at` | datetime | レコードの日付 | nursing_process, patient_condition, instruction |
| | `soap_s` / `soap_o` / `soap_a` / `soap_p` | text | 自由記述フィールド（S(主観)/O(客観)/A(評価)/P(計画)） | |
| | `recorder_name` | recorder | 記録者 | |
| `infections` | `name` | item(label=感染症) | フィールド `感染症: <値>`（患者横断＝サマリヘッダ行き） | risks |
| | `detail` | text | フィールド `内容: <値>` | |

`role` の語彙: `datetime`（日付軸）/ `item`（項目名・コード解決可）/ `value`（値）/ `unit`（単位）/ `text`（自由記述）/ `recorder`（記録者）/ `subtype`（区分・転帰）/ `phi`（個人識別情報＝出力除外）/ `id`・`link`（結合キー＝非出力）。

### 1.2 代替経路: POST /ask（整形済みテキスト）

`AskRequest`（`app.py`）で日付チャンク Markdown を直接渡す。前処理済みファイルや動作確認に使う。

| フィールド | 型 | 必須 | デフォルト | 説明 |
|---|---|---|---|---|
| `context` | `str` | 必須 | なし | `min_length=1`。医療記録テキスト（日付チャンク Markdown） |
| `patient_id` | `str` | 任意 | `"unknown"` | 患者ID |
| `template_id` | `str \| None` | 任意 | `None` | 省略時 `HOSPITAL` |

### 1.3 context のテキスト形式（両経路共通の契約）

`context` は `# 患者ID:` ＋ `## サマリ基本情報`（患者横断情報）＋ `- YYYYMMDD` 日付チャンクからなる Markdown。`ingest` ノードがこれを `chunks` / `grep_index` / `summary_header` に分解する（§3.2）。

```
# 患者ID: P1

## サマリ基本情報（全期間共通）
- 感染症
    感染症: MRSA 陽性
    内容: 接触予防継続

- 20230209
  - バイタルサイン
    体温: 37.8℃
    SpO2: 94%
    （記録者: 看護師A）
  - 看護記録
    S(主観):
      息苦しい
```

---

## 2. DB入力パイプライン（各項目 → context）

`build_context_from_db(patient_id, encounter_id, spec_id)`（`adapters/pipeline.py`）が各段を順に実行する。

| 順 | 段 | ファイル | 処理 |
|---|---|---|---|
| 1 | ロード・検証 | `query_specs_loader/loader.py` | `query_spec` を Pydantic 検証。`record_category` が `RecordCategory` メンバか、`retrieval.sql` のバインド変数が `:patient_id` / `:encounter_id` のみかを検査 |
| 2 | 取得 | `adapters/sql_source.py` | `get_adapter("sql")` で取得アダプタを生成し、各 `record` の SQL を `:patient_id` / `:encounter_id` バインドで実行。`{record_category: [行]}` を返す |
| 3 | 正規化 | `adapters/normalizer.py` | 各行の各カラムを `role` で `ClinicalRecord` フィールドへ写像。`role=item`+`codesystem` はコード→名称解決（未解決は `[未解決コード:...]`）。患者横断カテゴリ（`allergy`/`infection`/`nursing_problem`/`patient_profile`/`nursing_acuity`）は `cross_cutting=True`。取得0件カテゴリは `missing_categories` |
| 4 | サンプリング | `adapters/sampler.py` | `sampling.strategy=extremes` のカテゴリを項目ごとに first/last（日付）＋ min/max（数値）へ間引く |
| 5 | PHIマスク | `adapters/phi_masker.py` | 自由記述（`is_text`）の電話番号・郵便番号・メール等を非可逆マスク。`role=phi` 列は段3で既に除外 |
| 6 | Markdown化 | `adapters/markdown_renderer.py` | `# 患者ID:` ＋ サマリヘッダ（cross_cutting・欠損）＋ `- YYYYMMDD` 日付チャンクへ直列化。擬似日付 `00000000` は使わない |

出力 `context` は `AskRequest.context` と同形式であり、以降の処理（§3）は両経路共通。

---

## 3. 処理フロー（LangGraph v2）

`_run_graph` が `GlobalState`（`graph/state.py`）初期値を構築し、`build_nursing_summary_graph()`（`graph/builder.py`）でコンパイルしたグラフを `invoke` する。

```
START
  → ingest
  → [route_and_fanout]  ── total_tokens で分岐
       ≤ SINGLE_PASS_TOKEN_THRESHOLD(既定32768) → single_pass
       > 閾値                                   → section_worker(Send × セクション数)
  single_pass
  → [after_single_pass]  ── 本文が空のセクションのみ section_worker、無ければ assemble
  section_worker  ──(reducer merge_sections で section_results 集約)──→ assemble
  assemble → consistency → finalize → END
```

### 3.1 ノード別仕様

| ノード | ファイル | LLM | 主な出力 |
|---|---|---|---|
| `ingest` | `graph/nodes/ingest.py` | なし | `template`, `routing`, `summary_header`, `chunks`, `grep_index`, `total_tokens` |
| `single_pass` | `graph/nodes/single_pass.py` | あり（1回） | `section_results`（全セクション） |
| `section_worker` | `graph/nodes/section_worker.py` | あり | `section_results`（1セクション） |
| `assemble` | `graph/nodes/assemble.py` | なし | `draft_summary` |
| `consistency` | `graph/nodes/consistency.py` | あり（claim分解のみ） | `review_flags`（未支持claim） |
| `finalize` | `graph/nodes/finalize.py` | なし | `final_summary`, `review_flags` |

### 3.2 処理内容（要点）

- `ingest`: `load_template` / `load_routing`、`_extract_summary_header` で `summary_header`、`_build_search_index` で `chunks`、`explode_to_spans` で `grep_index`（`{date, category_label, text}`）、`tiktoken` で `total_tokens`。
- `route_and_fanout`（`graph/edges/routing_v2.py`）: `total_tokens ≤ 閾値` → `single_pass`、超 → 全セクションを `Send` で並列。
- `single_pass`: 全セクション + `summary_header` + 全 `chunks` を1プロンプトで `chat_json`（`think=False`, `num_ctx=LARGE_NUM_CTX`）。
- `section_worker`: `collect`（カテゴリ全件収集・top-k 制限なし＋ keyword grep）→ `_extract`（`chat_json`）→ 本文が空なら最大 `MAX_REFILL` 回再試行（**LLM スコア不使用**）。`absent_categories` で記録に無いカテゴリを欠落明示。
- `assemble`: `section_delimiter`（`--- {name} ---`）で決定論組立。空は `"記録なし（要確認）"`。
- `consistency`: `chat_json` で claim 分解→主要トークンの過半が `grep_index`＋`summary_header` に含まれるか決定論照合。未支持を `review_flags`（自動修復しない）。
- `finalize`: `final_summary = draft_summary`、`review_flag` 立ちセクションを集約。

### 3.3 検索ロジック（agentic search / `graph/search_index.py`）

#### agentic search とは

LLM を中核に据え、「①どの情報を集めるか計画 → ②検索ツールで記録を収集 → ③生成 → ④結果を点検」を必要に応じて反復する検索方式。単一の検索で固定的に上位 k 件を返す従来の検索（RAG の1パス検索）と異なり、収集・生成・点検を分けて制御できる。本システムは Web 検索ではなく**1患者の医療記録**を対象とするため、ベクトル DB を使わず grep（部分文字列マッチ）で実装する。

#### 本システムでの agentic search の構成

| agentic search の段階 | 本システムの実装 |
|---|---|
| ① 計画（どのカテゴリ・キーワードを集めるか） | テンプレートの routing（`<id>.routing.yaml`）。セクションごとに `categories` / `keywords` / `mode` を事前定義 |
| ② 収集（検索ツールで記録を集める） | `collect()` … カテゴリ全件収集（top-k 制限なし）＋ keyword grep（下記 (a)〜(c)） |
| ③ 生成 | `single_pass` / `section_worker._extract`（LLM が本文を生成） |
| ④ 点検 | `absent_categories`（記録に無いカテゴリを検出）、`consistency`（生成後に記録と照合） |
| 反復 | `section_worker` は本文が空なら最大 `MAX_REFILL` 回、`single_pass` は欠損セクションのみ `section_worker` へ。**反復はすべて決定論カウンタで制御し、LLM スコアは使わない** |

以下は段階②（収集）の詳細。**ベクトル DB を使わず、grep（部分文字列マッチ）と決定論的なカテゴリ照合のみ**で行う。収集処理自体は LLM を使わない。

#### (a) grep 索引の構築 — `explode_to_spans(chunks, chunk_index)`（`ingest` で実行）

各日付チャンクを `  - {カテゴリ名}` 小見出し単位のスパンに分解し、`grep_index = [{date, category_label, text}]` を構築する。小見出しが無いチャンク（テキスト/XML 由来）は `category_label=None` の単一スパンになる。

```
- 20230209
  - バイタルサイン       ┐ span{date:20230209, category_label:"バイタルサイン",
    体温: 37.8℃          │       text:"  - バイタルサイン\n    体温: 37.8℃\n    SpO2: 94%"}
    SpO2: 94%            ┘
  - 看護記録            ┐ span{date:20230209, category_label:"看護記録", text:...}
    S(主観): 息苦しい    ┘
```

#### (b) routing 定義 — `templates/<id>.routing.yaml`

セクションごとに「どのカテゴリ・キーワードを集めるか」「収集モード」を定義する。`categories` は `RecordCategory` の enum 値で書き、`resolve_category_labels()` が `CATEGORY_LABELS` で日本語ラベルへ解決してから grep する（grep のアンカーはラベル）。

```yaml
medical_equipment:                    # セクション
  mode: extractive
  categories: [procedure, medication, vital_sign]   # → 処置・医療機器 / 薬剤・服薬 / バイタルサイン
  keywords: ["カテーテル", "ドレーン", "挿入", "装着", "酸素", "点滴"]
```

#### (c) 収集 — `collect(entry, grep_index, chunks)`

| mode | 収集方法 |
|---|---|
| `extractive` | **① カテゴリ全件収集**: routing の `categories` をラベル解決し、`grep_index` の該当ラベルスパンを**全件**回収（**top-k で切らない**＝取りこぼし防止）。**② keyword grep**: `keywords` の正規表現に一致するスパンを補完（カテゴリ越境・`category_label=None` スパンも対象）。 |
| `synthetic` | 全日付チャンクを供給（`nursing_process` / `risks` 等。検索で絞らず時系列全体を渡す）。 |

戻り値は `(収集テキスト, 実在カテゴリラベルの集合 present_labels, 関与日付)`。

```
collect("medical_equipment") の例:
  ① 全件収集: category_label ∈ {処置・医療機器, 薬剤・服薬, バイタルサイン} のスパンを全件
  ② keyword: 本文に「酸素」「点滴」等を含むスパンを追加（重複除外）
  → これらを evidence として section_worker の _extract に渡す
```

#### (d) 欠落検出 — `absent_categories(entry, present_labels)`

routing が要求するが記録に存在しないカテゴリ（`target_labels − present_labels`）を返す。記録自体に無いカテゴリは grep しても得られないため、`section_worker` は欠落として `missing` に記録し `review_flag` を立てる（assemble で「記録なし（要確認）」、finalize で `review_flags`）。

#### 設計上の要点

- **検索ヒット依存からの脱却**: カテゴリ全件収集が主経路（該当カテゴリの行は全て渡す）。keyword grep はカテゴリ越境・未分類の補完。`max_results` のような件数上限を設けない。
- **患者横断情報の常時供給**: `summary_header`（アレルギー・感染症・看護問題等、§1.3）は検索に依存せず `single_pass` / `_extract` に常に渡る。
- **決定論**: 収集・欠落判定は LLM・スコアを使わない純関数（`graph/search_index.py`）。

### 3.4 LLM 呼び出し（`llm/client.py`）

- `chat(prompt, format_schema, temperature, think=False, options_override)`: `think=False` で reasoning 混入防止、`options_override` で `num_ctx` 上書き。
- `chat_json(prompt, schema, max_retries=3)`: `chat` + `extract_json`（コードフェンス除去・平衡括弧抽出）+ retry。
- モデル/接続先は `ENV` 切替（production=ローカル `gpt-oss:120b`、test=`gpt-oss:120b-cloud`）。

---

## 4. 出力（`app.py`: AskResponse）

| フィールド | 型 | 説明 |
|---|---|---|
| `answer` | `str` | `final_summary`。`section_delimiter` 区切りの看護サマリー |
| `template_id` | `str` | 使用テンプレートID |
| `review_flags` | `list[str]` | 人手レビュー対象（欠落セクションキー・未支持 claim） |

`answer` はテンプレート（`templates/<id>.yaml`）の `section_delimiter` で区切られる。`hanwa` は6セクション、`shinkinen` は2セクション。

---

## 5. GlobalState（`graph/state.py`）

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
