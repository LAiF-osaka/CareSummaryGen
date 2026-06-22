# システムアーキテクチャ（v2）

本書は CareSummaryGen のシステムアーキテクチャ全体像を示す。詳細は各設計書を参照する。

- 実行時フロー（入力→処理→出力）: [data-flow.md](data-flow.md) / [data-flow-report.html](data-flow-report.html)
- agentic search の設計: [agentic-search-redesign.md](agentic-search-redesign.md)
- DB入力の設計: [db-input-design.md](db-input-design.md)
- コーディング規約: [.claude/rules/coding-style.md](../.claude/rules/coding-style.md)

比喩・平易な言い換え・抽象的な言い換えは用いず、関数名・フィールド名・ファイル名で記述する。

---

## 1. 技術スタック

| レイヤー | 採用 |
|---|---|
| オーケストレーション | LangGraph（`StateGraph` + `Send` 並列） |
| LLM 呼び出し | Ollama Python SDK（`llm/client.py`。LangChain 不使用） |
| LLM モデル | gpt-oss:120b（`ENV=production`=ローカル / `ENV=test`=Ollama Cloud） |
| 構造化出力 | Pydantic スキーマ + `chat_json`（format + extract_json + retry） |
| Web API | FastAPI（`/ask`, `/ingest`, `/templates`, `/`） |
| DB取得 | SQLAlchemy（`adapters/sql_source.py`、query_spec 駆動） |
| トークン計算 | tiktoken（cl100k_base） |
| 依存管理 | uv |
| 整形・静的解析 | Black(79) / flake8 / mypy / isort |

---

## 2. コンポーネント構成

```
┌─────────────────────────────────────────────────────────────────────┐
│ API 層 (app.py / FastAPI)                                            │
│   POST /ask（テキスト入力）   POST /ingest（DB入力）                  │
│   _run_graph(context, patient_id, template_id) に収束               │
└───────────────┬──────────────────────────┬──────────────────────────┘
                │                          │ build_context_from_db()
                │                          ▼
                │            ┌─────────────────────────────────────────┐
                │            │ DB入力パイプライン (adapters/)            │
                │            │  query_specs_loader → sql_source.fetch   │
                │            │  → normalizer → sampler → phi_masker     │
                │            │  → markdown_renderer → context           │
                │            └─────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ オーケストレーション層 (graph/ / LangGraph v2)                       │
│   ingest → [route_and_fanout]                                        │
│     ≤閾値 → single_pass → [after_single_pass] → section_worker?      │
│     >閾値 → section_worker(Send × N)                                 │
│   → assemble → consistency → finalize → END                         │
│   検索: graph/search_index.py（grep 索引・カテゴリ全件収集・決定論）      │
└───────────────┬──────────────────────────────────────────────────────┘
                ▼
┌─────────────────────────────────────────────────────────────────────┐
│ LLM 層 (llm/client.py / Ollama SDK)                                  │
│   chat() / chat_json() / extract_json()  think=False, options_override│
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. ディレクトリ構成

| パス | 役割 |
|---|---|
| `app.py` | FastAPI エントリーポイント（`/ask`, `/ingest`） |
| `config/settings.py` | 環境変数・定数（`ENV`, `SINGLE_PASS_TOKEN_THRESHOLD`, `LARGE_NUM_CTX`, `MAX_REFILL`, `LLM_OPTIONS`） |
| `llm/client.py` | Ollama クライアント、`chat`/`chat_json`/`extract_json` |
| `llm/prompts.py` | single_pass / section_extract / consistency プロンプト |
| `graph/state.py` | `GlobalState` / `SectionResult` / reducer |
| `graph/builder.py` | v2 グラフ構築 |
| `graph/search_index.py` | grep 索引（`explode_to_spans` / `collect` / `absent_categories`） |
| `graph/nodes/` | `ingest`, `single_pass`, `section_worker`, `assemble`, `consistency`, `finalize` |
| `graph/edges/routing_v2.py` | `route_and_fanout`, `after_single_pass` |
| `templates/` | `<id>.yaml`（セクション定義）, `<id>.routing.yaml`（カテゴリ routing） |
| `templates_loader/` | `loader.py`（テンプレート）, `routing.py`（routing 検証・enum→ラベル解決） |
| `adapters/` | DB入力層（models, base, sql_source, normalizer, sampler, phi_masker, markdown_renderer, pipeline） |
| `query_specs/` | `<id>.yaml`（取得仕様）, `codesystems/<id>.yaml`（コード→名称） |
| `query_specs_loader/` | query_spec の Pydantic 検証・バインド変数検証 |
| `utils/` | 前処理（XML/zip → Markdown） |
| `tests/` | ユニット + モック E2E + 実 Ollama E2E + DB SQLite E2E |
| `docs/` | 本書・各設計書・データフロー仕様 |

---

## 4. 設計上の確定事項

- **検索はハイブリッド（決定論収集 ＋ LLM補完検索）**。ベクトル DB を使わず、`grep_index`（`{date, category_label, text}`）に対しカテゴリ全件収集（top-k 制限なし）＋ keyword grep で決定論収集（安全網）し、section_worker 内で LLM が不足を判断して追加検索クエリを動的生成する補完ループ（agentic 部分・`MAX_SEARCH_STEPS` 上限）を回す。正準的な agentic search（検索を全て LLM が駆動）ではなく、gpt-oss 信頼性と医療網羅性のエビデンスに基づくハイブリッド（[agentic-search-redesign.md](agentic-search-redesign.md) 参照）。
- **網羅性は決定論ガードレールで保証**。`collect()` の全件収集が補完ループに先行（安全網）、`absent_categories` が後行で記録に無いカテゴリを欠落明示（LLM の停止判断に依存しない）。`chat_json` 失敗時は補完ループを飛ばし決定論モードへフォールバック。
- **入力規模で経路を分ける**。`total_tokens ≤ SINGLE_PASS_TOKEN_THRESHOLD` は single_pass（1回生成）、超はセクション並列。
- **テンプレートは YAML で外部化**。新機関は `templates/<id>.yaml` + `<id>.routing.yaml` の追加のみで対応（コード変更不要）。
- **DB入力はカラム未確定でも動く**。`query_spec` を論理層（role）と retrieval 層（sql 等）に分離。`retrieval.sql` のバインド変数は `:patient_id` / `:encounter_id` のみ許可。
- **PHI**: 構造化列は `role=phi` で除外、自由記述は regex で非可逆マスク。閉域運用前提で認証は未実装。

---

## 5. Definition of Done

- 既存動作が壊れていないこと（テストで証明）
- コードレビュー完了（Critical ゼロ）
- 新規・変更コードに docstring・型アノテーション（[coding-style.md](../.claude/rules/coding-style.md)）
- Black(79) / flake8 クリーン
