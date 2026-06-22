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
- `section_worker`: `collect`（決定論的全件収集）→ `supplemental_search`（**観測駆動の agentic ループ**: LLM が収集状況を観測し不足を同定→追加クエリを動的生成→言い換え、決定論ガードレールで停止・最大 `MAX_SEARCH_STEPS`）→ `_extract`（`chat_json`、本文が空なら最大 `MAX_REFILL` 回再試行）→ `absent_categories` で欠落明示（§3.3 ハイブリッド）。
- `assemble`: `section_delimiter`（`--- {name} ---`）で決定論組立。空は `"記録なし（要確認）"`。
- `consistency`: `chat_json` で claim 分解→主要トークンの過半が `grep_index`＋`summary_header` に含まれるか決定論照合。未支持を `review_flags`（自動修復しない）。
- `finalize`: `final_summary = draft_summary`、`review_flag` 立ちセクションを集約。

### 3.3 検索ロジック（ハイブリッド: 決定論収集 ＋ LLM補完検索 / `graph/search_index.py`）

#### 方式と用語の正確化

本システムの検索は **ハイブリッド** である。正準的な agentic search（実行時に LLM が全ての検索クエリ生成・評価・停止を駆動する。Anthropic の定義で agent）ではなく、**決定論的な全件収集を骨格（安全網）とし、その内側に LLM 駆動の補完検索ループ（agentic 部分）を、決定論ガードレールに囲んで持つ**構成である。この選択は gpt-oss の信頼性と医療の網羅性の2軸のエビデンスに基づく（[agentic-search-redesign.md](agentic-search-redesign.md) の「実装監査と用語修正」を参照）。

| 段階 | 実装 | LLM | 性質 |
|---|---|---|---|
| ① 決定論収集 | `collect()` … カテゴリ全件収集（top-k 制限なし）＋ keyword grep | なし | 安全網（必須カテゴリを悉皆で確保） |
| ② LLM 補完検索ループ | `supplemental_search()` … 観測駆動の manual ReAct ループ。LLM が収集状況（カバレッジ・検索履歴）を観測→不足を同定→追加クエリ（keyword / category / date_range）を**動的生成**→言い換え、grep で実行、新規スパンを追加 | あり | agentic（不足同定・クエリ生成・停止判断を LLM が観測に基づき行う。grep 実行は決定論） |
| ③ 生成 | `single_pass` / `section_worker._extract`（LLM が本文を生成） | あり | — |
| ④ 決定論点検 | `absent_categories`（記録に無いカテゴリを検出）、`consistency`（生成後に記録と照合） | なし(absent)/あり(claim分解) | 決定論ガードレール |

**決定論ガードレール**（②の暴走・gpt-oss 破損を遮断・**停止の最終決定権は決定論側**で LLM の `need_more` は助言）: ハード反復上限 `MAX_SEARCH_STEPS`、進捗ゼロ（新規スパン0）連続2回で停止、同一クエリ繰り返し検出、`collect()` の全件収集が②に先行（安全網）、`absent_categories` が②の後で網羅点検（LLM の停止判断に依存しない）。`chat_json` が失敗（None）すれば②を飛ばして①のみで続行する（グレースフル・フォールバック＝決定論モード）。各ステップの観測・クエリ・停止理由は `SectionResult.search_trace` に記録（監査可能性）。

`single_pass`（≤閾値）は全チャンクを供給するため②は不要。②は section_worker（>閾値 / 欠損セクション）のみに適用する。

以下は段階①（決定論収集）の詳細。**ベクトル DB を使わず、grep（部分文字列マッチ）と決定論的なカテゴリ照合のみ**で行う。①自体は LLM を使わない。

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

#### (e) LLM 補完検索ループ — `supplemental_search()`（section_worker 内・本物の agentic search）

①の決定論収集は routing に静的固定された categories/keywords しか集めない。これだけでは**語彙のミスマッチ**（同義語・略語・言い換え）、**含意**（降圧薬名から高血圧の記録を引く等）、**カテゴリ越境**（リスク情報が看護記録 S 欄に埋まる等）で取りこぼしが生じる（EHR エンティティ検索の実測で含意マッチ BM25 MRR=36.30%、完全一致 83.92% と47pt差: [arXiv:2502.06252](https://arxiv.org/html/2502.06252v1)）。②はこの穴を**実行時の LLM 推論で埋める**。LLM は「収集状況を観測 → 不足を同定 → 不足を埋める検索を1手指示 → 結果を観測 → 言い換え」を反復する（ReAct ループ）。これが「飾り」でなく本物の agentic search である根拠を、各ステップの観測材料・動的判断・決定論ガードレールで担保する。

**観測材料（毎ステップ LLM に再投入。ReAct の Observation）**:
- **セクション目標**: `name` / `description`。
- **収集済みカバレッジ**: `collected` を `category_label` で集計した「カテゴリ別件数・収集済み日付範囲」（決定論計算）。LLM が「どこが薄いか」を見る。
- **検索履歴**: 直近3手の `{tool, query, 追加件数 / ゼロ件}`。
- **既出クエリ集合**: 同一クエリの再発行を禁止（言い換えを促す）。
- **記録全体の地図**: 検索可能な全カテゴリ・全日付。

```python
for step in range(MAX_SEARCH_STEPS):                  # ハード上限（フェイルセーフ）
    coverage = _coverage_report(collected, grep_index)        # 観測: カバレッジ（決定論集計）
    history  = _format_history(trace)                         # 観測: 直近の検索結果
    decision = chat_json(SUPPLEMENT_PROMPT(目標, coverage, history, 既出クエリ, 地図), SCHEMA)
    # decision = {"satisfied_points": [...], "missing_points": [...],  ← gap 分析（充足/不足の同定）
    #             "need_more": bool, "tool": "keyword"|"category"|"date_range",
    #             "keyword"/"category"/"start_date"/"end_date", "reason": "..."}
    if decision is None:                              # 構造化失敗 → 決定論へフォールバック
        break
    if not decision["need_more"] and not decision["missing_points"]:   # gap 駆動の充足判定で停止
        break
    if query in tried_queries:                        # 繰り返し検出 → 言い換え促し（即停止しない）
        no_progress += 1; (no_progress>=2 で停止); continue
    new   = execute_search_tool(decision.tool, decision, grep_index)   # grep 実行（決定論）
    added = [s for s in new if s not in seen]
    trace.append(観測)                                # 監査トレース
    if not added:                                     # ゼロ進捗 → 即停止せず言い換えの機会（R4）
        no_progress += 1; (no_progress>=2 で停止); continue
    no_progress = 0; collected += added; seen |= set(added)
```

**停止条件（決定論 OR・LLM の `need_more` は助言）**: ① gap 充足（`missing_points` 空かつ `need_more=false`）/ ② ハード反復上限 `MAX_SEARCH_STEPS`（`for` で物理保証）/ ③ 進捗ゼロ連続2回（no-progress 検出）/ ④ `chat_json` None（フォールバック）。同一クエリ繰り返しは即停止せず1手スキップして言い換えを促す（reformulation）。

検索ツール: `keyword`（同義語・略語・関連語で部分文字列 grep）、`category`（routing 外カテゴリを LLM が動的指定して全件回収＝カテゴリ越境対策）、`date_range`（特定期間の取りこぼし補完）。実行は全て決定論（クエリは LLM・実行は決定論）。

#### 設計上の要点（「本物の agentic search」受入基準）

agentic search が「飾り」でなく本物である条件を、実装の受入基準として固定する（[Firecrawl / Search Engine Land 2026-01](https://searchengineland.com/beyond-rag-ai-search-agentic-content-478996)、[Anthropic 2025-12](https://www.anthropic.com/research/building-effective-agents)、[SIM-RAG arXiv:2505.02811](https://arxiv.org/html/2505.02811v1)）。

| # | 受入基準 | 実装 |
|---|---|---|
| 1 | **観測ループが閉じる**: 直前アクションの結果が次の LLM 入力に入る | `coverage` / `history` を毎ステップ注入（ReAct Observation） |
| 2 | **観測駆動の再定式化**: 結果に応じてクエリが変わる・繰り返さない | 既出クエリ集合の注入＋ゼロ件で即停止せず言い換え |
| 3 | **gap 駆動の停止**: 不足を同定してから停止 | `missing_points` を列挙させ、空で充足停止 |
| 4 | **決定論ガードレールが LLM 停止を囲う** | 反復上限・no-progress・フォールバックの OR |
| 5 | **実行時に LLM が何を探すか決める** | `missing_points` と次クエリを観測から動的生成（routing 静的固定を補完） |
| 6 | **トレースが残る（監査可能性）** | `SectionResult.search_trace` に観測・クエリ・停止理由を記録 |

- **ハイブリッド（guarded agent）**: ①決定論収集（安全網・下限保証）＋②agentic 補完（上乗せ・破壊不能）＋④決定論点検。②の LLM は網羅を**上乗せ改善するだけで破壊できない**。前段 `collect()` 全件収集と後段 `absent_categories()` が LLM の挙動に依存せず網羅の下限と欠落明示を保証する（verifier-in-the-loop。医療の高ステークス要件: [EU AI Act 第14条](https://artificialintelligenceact.eu/article/14/)）。
- **検索ヒット依存からの脱却**: カテゴリ全件収集が主経路（該当カテゴリの行は全て渡す）。`max_results` のような件数上限を設けない。②はそれを上乗せ補完するのみで、①の網羅を損なわない。
- **患者横断情報の常時供給**: `summary_header`（アレルギー・感染症・看護問題等、§1.3）は検索に依存せず `single_pass` / `_extract` に常に渡る。
- `single_pass`（≤閾値）は全チャンク供給のため②は不要。②は section_worker のみに適用する。

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
