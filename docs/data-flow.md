# データフロー仕様: 入力 → 処理 → 出力

本書は CareSummaryGen の `POST /ask` における入力データ、処理フロー、出力データを、
実装上の関数名・State フィールド名・プロンプト名に対応づけて記述する。

対象コミット時点の構成: LangGraph + Ollama Python SDK + FastAPI / Agentic Search パイプライン。

---

## 1. 入力

### 1.1 HTTP リクエスト

- メソッド: `POST`
- パス: `/ask`
- Content-Type: `application/json`
- 定義: `app.py` の `AskRequest`（Pydantic モデル）

| フィールド | 型 | 必須 | デフォルト | 制約・説明 |
|---|---|---|---|---|
| `context` | `str` | 必須 | なし | `min_length=1`。医療記録テキスト本体 |
| `patient_id` | `str` | 任意 | `"unknown"` | 患者識別子 |
| `template_id` | `str \| None` | 任意 | `None` | `None` の場合は環境変数 `HOSPITAL`（既定 `hanwa`）が使用される |

### 1.2 `context` のテキスト形式

`graph/nodes/input_adapter.py` の `_build_search_index()` は、
正規表現 `^- (\d{8})\s*$`（行頭の `- YYYYMMDD`）をチャンク境界として分割する。
この日付行が1つ以上存在する場合、日付単位でチャンクが生成され、
各チャンクに `{"date": "YYYYMMDD", "index": i}` のメタデータが付与される。

日付行が存在しない場合は `_split_by_tokens()` にフォールバックし、
`tiktoken` の `cl100k_base` でトークン分割する
（`chunk_size = min(CHUNK_SIZE, 4096)`、`overlap = min(CHUNK_OVERLAP, 200)`、
各チャンクのメタデータは `{"date": "unknown", "index": i}`）。

### 1.3 入力例

```json
{
  "context": "# 患者ID: DEMO001\n\n- 20240115\n  - カルテ#1\n    主訴: 発熱と呼吸困難\n    バイタルサイン: BP 142/88 mmHg、HR 96/分、BT 38.6℃、SpO2 91%\n    酸素投与開始: 鼻カニューレ 2L/分\n\n- 20240116\n  - カルテ#1\n    抗菌薬投与開始: セフトリアキソン 2g 1日1回 点滴静注\n    血液検査: WBC 14200/μL、CRP 18.5 mg/dL\n",
  "patient_id": "DEMO001",
  "template_id": "hanwa"
}
```

上記の `context` は `- 20240115` と `- 20240116` の2つの日付行を含むため、
2チャンクに分割され、`chunk_index` は
`[{"date": "20240115", "index": 0}, {"date": "20240116", "index": 1}]` となる。

---

## 2. 処理フロー

`POST /ask`（`app.py` の `ask()`）は `NursingSummaryState`（`graph/state.py`）の初期値を構築し、
`build_nursing_summary_graph()`（`graph/builder.py`）でコンパイルした LangGraph グラフを `invoke` する。

ノードとエッジの接続は次のとおり。

```
START
  → input_adapter
  → plan ──→ search ──→ extract ──→ evaluate
                ↑                      │
                │  evaluate_sufficiency │ (条件分岐)
                └──────────────────────┤
                                       ├─ "search"        → search へ戻る
                                       └─ "next_section"  → next_section
  next_section ──(has_more_sections)──┬─ "plan"       → plan へ戻る
                                       └─ "synthesize" → synthesize
  → synthesize
  → reflect ──(should_continue_reflection)──┬─ "revise"           → revise → reflect へ戻る
                                            └─ "output_formatter" → output_formatter
  → output_formatter
  → END
```

### 2.1 ノード別仕様

| 順 | ノード | ファイル | LLM 呼出 | 入力フィールド | 出力フィールド |
|---|---|---|---|---|---|
| 1 | `input_adapter` | `graph/nodes/input_adapter.py` | なし | `template_id`, `raw_context` | `template`, `search_plan`, `chunks`, `chunk_index`, `current_section_idx=0`, `search_iteration=0`, `section_results={}`, `_search_results=[]` |
| 2 | `plan` | `graph/nodes/plan.py` | あり | `search_plan`, `current_section_idx`, `chunk_index` | `search_plan`（当該セクションの `search_queries` 更新）, `search_iteration=0` |
| 3 | `search` | `graph/nodes/search.py` | あり（ツール呼出） | `search_plan`, `current_section_idx`, `chunks`, `chunk_index` | `_search_results` |
| 4 | `extract` | `graph/nodes/extract.py` | あり | `_search_results`, `search_plan`, `current_section_idx` | `section_results`（当該 `section_key` に追記）, `search_iteration += 1` |
| 5 | `evaluate` | `graph/nodes/evaluate.py` | あり | `section_results`, `search_plan`, `current_section_idx` | `_section_sufficient`, （不足時）`search_plan` の `search_queries` 更新 |
| - | `next_section` | `graph/edges/search_loop.py` | なし | `current_section_idx` | `current_section_idx += 1`, `search_iteration=0`, `_search_results=[]`, `_section_sufficient=False` |
| 6 | `synthesize` | `graph/nodes/synthesize.py` | あり | `template`, `section_results` | `draft_summary` |
| 7 | `reflect` | `graph/nodes/reflect.py` | あり | `template`, `draft_summary`, `iteration_count` | `reflection_feedback`, `reflection_approved`, `iteration_count += 1` |
| 8 | `revise` | `graph/nodes/revise.py` | あり | `draft_summary`, `reflection_feedback` | `draft_summary`（更新） |
| 9 | `output_formatter` | `graph/nodes/output_formatter.py` | なし | `draft_summary` | `final_summary` |

### 2.2 各ノードの処理内容

#### 1. input_adapter

- `load_template(template_id)` でテンプレート定義（YAML）をロードする。
- テンプレートの `sections` から `search_plan` を初期化する。各要素は
  `{"section_key", "section_name", "description", "search_queries": []}`。
- `_build_search_index(raw_context)` で `chunks` と `chunk_index` を生成する（1.2 参照）。

#### 2. plan

- `search_plan[current_section_idx]` を対象セクションとする。
- `chunk_index` から日付の集合を取得し `available_dates` を構成する。
- `PLAN_PROMPT`（`llm/prompts.py`）に `section_name`, `section_description`, `available_dates` を埋め込み、
  `chat(prompt, format_schema={"queries": [str]}, temperature=0.0)` を呼ぶ。
- 応答 JSON を `json.loads` して `queries` を取得する。
  パースに失敗した場合は `_extract_keywords_from_text()` にフォールバックし、
  箇条書き行（長さ50未満）を最大5件まで抽出する。
- 当該セクションの `search_queries` に格納する。

#### 3. search

- 提供ツール: `search_by_keyword(keyword)`、`search_by_date_range(start_date, end_date)`。
- `chat_with_tools(SEARCH_PROMPT, tools=[...])` を呼び、応答の `tool_calls` を実行する。
  - `search_by_keyword`: `_execute_keyword_search()` が `keyword.lower()` を含むチャンクを返す。
  - `search_by_date_range`: `_execute_date_range_search()` が `start_date <= meta["date"] <= end_date` のチャンクを返す。
- `tool_calls` が存在しない、または例外が発生した場合は、
  当該セクションの `search_queries` を用いて `_execute_keyword_search()` でフォールバック検索する。
- 結果は `hash` ベースで重複除去し、先頭5件を `_search_results` とする。

#### 4. extract

- `_search_results` が空の場合、`section_results[section_key] = "記録なし"` とし、`search_iteration += 1`。
- 空でない場合、`EXTRACT_PROMPT` に `section_name`, `section_description`,
  および `_search_results` を `\n\n---\n\n` で結合した文字列を埋め込み、`chat(prompt)` を呼ぶ。
- 応答を `section_results[section_key]` に格納する。`section_results` は
  `merge_dicts` reducer（`graph/state.py`）により、同一キーへは改行区切りで追記される。
- `search_iteration += 1`。

#### 5. evaluate

- `EVALUATE_PROMPT` に `section_name`, `section_description`,
  当該セクションの抽出済みテキストを埋め込み、`chat(prompt, temperature=0.0)` を呼ぶ。
- 応答に `"SUFFICIENT"`（大文字化して判定）が含まれれば `_section_sufficient=True`。
- 含まれない場合、`_extract_additional_queries()` で応答から追加検索クエリを抽出し、
  当該セクションの `search_queries` を更新、`_section_sufficient=False`。
- 条件分岐 `evaluate_sufficiency()`（`graph/edges/search_loop.py`）:
  - `_section_sufficient == True` または `search_iteration >= max_search_iterations` → `next_section`
  - それ以外 → `search`（再検索）

#### next_section（エッジ関数）

- `current_section_idx += 1`、`search_iteration=0`、`_search_results=[]`、`_section_sufficient=False`。
- 条件分岐 `has_more_sections()`:
  - `current_section_idx < len(search_plan)` → `plan`（次セクションへ）
  - それ以外 → `synthesize`

#### 6. synthesize

- `build_format_instruction(template)`（`templates_loader/loader.py`）で
  テンプレートのセクション定義からフォーマット指示文を生成する。
- `template["sections"]` の各セクションについて、
  `section_results` の該当値（未取得時は `"記録なし"`）を `### {name}\n{content}` 形式で連結する。
- `SYNTHESIZE_PROMPT` に上記2つを埋め込み、`chat(prompt)` を呼ぶ。応答を `draft_summary` に格納する。

#### 7. reflect

- `template["sections"]` のセクション名一覧を構成する。
- `REFLECTION_PROMPT` に セクション名一覧と `draft_summary` を埋め込み、
  `chat(prompt, temperature=0.0)` を呼ぶ。
- 応答に `"APPROVED"`（大文字化して判定）が含まれれば `reflection_approved=True`。
  `chat` が例外を送出した場合も `reflection_approved=True` とする。
- `iteration_count += 1`。
- 条件分岐 `should_continue_reflection()`（`graph/edges/reflection.py`）:
  - `reflection_approved == True` → `output_formatter`
  - `iteration_count >= max_iterations` → `output_formatter`
  - それ以外 → `revise`

#### 8. revise

- `REVISION_PROMPT` に `draft_summary` と `reflection_feedback` を埋め込み、`chat(prompt)` を呼ぶ。
- 応答で `draft_summary` を更新し、`reflect` へ戻る。

#### 9. output_formatter

- `final_summary = draft_summary` とする。

### 2.3 LLM 呼び出し条件

`llm/client.py` の `chat()` / `chat_with_tools()` は `ollama_client.chat()` を呼ぶ。

- モデル: `config.settings.MODEL_NAME`（`ENV=production` で `gpt-oss:120b`、`ENV=test` で `gpt-oss:120b-cloud`）
- 接続先: `config.settings.OLLAMA_BASE_URL`（`ENV=production` で `http://localhost:11434`、`ENV=test` で `https://ollama.com`）
- オプション: `LLM_OPTIONS = {temperature: 0.1, top_p: 0.92, repeat_penalty: 1.2, num_ctx: 8192, num_predict: 4096}`
  （`plan` / `evaluate` / `reflect` は `temperature=0.0` で上書き）
- `keep_alive`: `"60m"`

### 2.4 ループの上限

| ループ | 制御変数 | 上限 | 既定値 |
|---|---|---|---|
| セクション内再検索（`search`→`extract`→`evaluate`→`search`） | `search_iteration` | `max_search_iterations` | `MAX_SEARCH_ITERATIONS = 3` |
| Reflection（`reflect`→`revise`→`reflect`） | `iteration_count` | `max_iterations` | `MAX_REFLECTION_ITERATIONS = 2` |

セクションのループは全 `search_plan` 要素を `current_section_idx` で順に処理する。

---

## 3. 出力

### 3.1 HTTP レスポンス

- 定義: `app.py` の `AskResponse`（Pydantic モデル）
- ステータス: 正常時 `200`、`template_id` 不正時 `400`、グラフ内エラー時 `500`

| フィールド | 型 | 説明 |
|---|---|---|
| `answer` | `str` | `final_summary`。テンプレート形式の看護サマリー |
| `template_id` | `str` | 使用されたテンプレート ID |
| `iteration_count` | `int` | Reflection の反復回数 |

### 3.2 `answer` の構造

`answer` はテンプレート（`templates/{template_id}.yaml`）の `sections` で定義された区切りを持つ。
`hanwa` テンプレートの `section_delimiter` は `"--- {name} ---"` であり、
6セクション（`指導した内容` / `医療機器装着・挿入・処置部位` / `入院中の看護の経過（生活状況）` /
`患者への病状説明及び本人・家族の受け止め方` / `継続される問題（今後のリスク）` / `その他`）で構成される。
`shinkinen` テンプレートは2セクション（`入院中の経過及び看護上の問題経過` / `備考`）。

### 3.3 出力例（構造）

```json
{
  "answer": "--- 指導した内容 ---\n[退院後の服薬指導・感染予防指導の内容]\n\n--- 医療機器装着・挿入・処置部位 ---\n[酸素投与・点滴に関する情報]\n\n--- 入院中の看護の経過（生活状況） ---\n2024年1月15日 [入院時の状態]\n2024年1月16日 [治療経過]\n...\n\n--- 患者への病状説明及び本人・家族の受け止め方 ---\n[病状説明と反応]\n\n--- 継続される問題（今後のリスク） ---\n[継続課題]\n\n--- その他 ---\n[その他事項]",
  "template_id": "hanwa",
  "iteration_count": 1
}
```

---

## 4. State フィールド一覧

`graph/state.py` の `NursingSummaryState`（TypedDict）。

| フィールド | 型 | 用途 |
|---|---|---|
| `patient_id` | `str` | 患者識別子 |
| `raw_context` | `str` | 入力医療記録テキスト |
| `hospital` | `str` | 環境変数 `HOSPITAL` の値 |
| `chunks` | `list[str]` | 分割済みチャンク本文 |
| `chunk_index` | `list[dict]` | 各チャンクのメタデータ（`date`, `index`） |
| `template_id` | `str` | テンプレート ID |
| `template` | `dict` | ロード済みテンプレート定義 |
| `search_plan` | `list[dict]` | セクション別の検索計画 |
| `section_results` | `dict[str, str]` | セクション別の抽出結果（`merge_dicts` reducer） |
| `current_section_idx` | `int` | 処理中セクションのインデックス |
| `search_iteration` | `int` | 当該セクションの検索反復回数 |
| `max_search_iterations` | `int` | 検索反復の上限 |
| `_search_results` | `list[str]` | `search` の結果（ノード間受け渡し） |
| `draft_summary` | `str` | 統合および改訂後のドラフト |
| `reflection_feedback` | `str` | `reflect` の応答テキスト |
| `reflection_approved` | `bool` | `reflect` が `APPROVED` を返したか |
| `iteration_count` | `int` | Reflection の反復回数 |
| `max_iterations` | `int` | Reflection 反復の上限 |
| `final_summary` | `str` | 最終出力 |
| `error` | `Optional[str]` | エラーメッセージ |
