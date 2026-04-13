# 看護サマリー生成システム — アーキテクチャ & 詳細設計書

## 1. エグゼクティブサマリー

本設計書は、CareSummaryGen の現行 Flask + LangChain 実装を **LangGraph + Ollama Python SDK + FastAPI** ベースに移行するための詳細設計を定義する。

### 技術選定

| レイヤー | 現行 | 移行後 |
|---|---|---|
| オーケストレーション | なし（手動 Map-Reduce） | **LangGraph** (StateGraph) |
| LLM 呼び出し | LangChain `Ollama.invoke()` | **Ollama Python SDK** `client.chat()` |
| 構造化出力 | なし（プレーンテキスト） | **Pydantic** + Ollama `format` パラメータ |
| Web API | Flask | **FastAPI**（async 対応・型安全・自動 OpenAPI ドキュメント） |
| トークン計算 | tiktoken (cl100k_base) | tiktoken（変更なし） |
| 依存管理 | uv | uv（変更なし） |

### LangGraph と LangChain の関係

```
pip install langgraph ollama
```

- `langgraph` は `langchain-core` に**ハード依存**する（自動インストールされる）
- しかし、ユーザコードで `langchain` や `langchain-community` を **import する必要はない**
- ノード関数は `state: dict → dict` の純粋な Python 関数
- ノード内で Ollama SDK を直接呼び出す

---

## 2. Agentic Search の導入根拠

### 2.1 Map-Reduce の限界

現行の Map-Reduce アプローチには以下の構造的限界がある:

| 問題 | 説明 |
|---|---|
| **盲目的チャンキング** | 固定トークン数で機械的に分割するため、臨床的な文脈（入院〜退院の流れ）が断絶する |
| **情報の希薄化** | 130,000 トークンのチャンクから抽出した中間要約を再度圧縮するため、重要な数値（バイタル・投薬量）が脱落しやすい |
| **クエリ非依存** | テンプレートのどのセクションに何が必要かを考慮せず、全情報を均一に抽出する |
| **一方通行** | Map → Reduce の1パスで完結し、情報不足を検知して再検索する手段がない |

### 2.2 Agentic Search とは

Agentic Search は、LLM エージェントが**検索ツールを自律的に操作して**情報を反復的に収集するアプローチである。
Agentic RAG（ベクトル DB による類似度検索を中心とする）とは異なり、
Agentic Search は**検索をアクティブな推論プロセスとして**扱う。

> "RAG treats retrieval as a preprocessing step; agentic search treats it as an active reasoning process."
> — Morph LLM, "Agentic Search: How Coding Agents Find the Right Code"

**Agentic Search の定義（arXiv:2602.17518 + Chroma Docs + Morph LLM より）:**

1. **LLM がツールを呼び出して検索する**: grep, キーワード検索, 日付範囲フィルタ等の検索ツールを LLM が自律的に選択・実行する
2. **反復的**: 1回の検索で終わらず、Plan → Search → Evaluate → Iterate のループを回す
3. **推論駆動**: 検索結果を評価し、不足があればクエリを書き換えて再検索する（Chain-of-Thought → Search → Evaluate）
4. **適応的**: 検索戦略を動的に変更できる（キーワード変更、日付範囲変更、検索対象の絞り込み等）

**Agentic Search ≠ Agentic RAG:**

| 項目 | Agentic RAG | Agentic Search |
|---|---|---|
| 検索手法 | ベクトル埋め込み + 類似度検索 | ツール呼び出し（grep, キーワード, フィルタ） |
| インデックス | 事前構築が必要（ベクトル DB） | 不要（テキストを直接検索） |
| 検索の主体 | パイプラインが自動実行 | **LLM エージェントが自律的に判断・実行** |
| 外部依存 | ベクトル DB（ChromaDB 等） | なし（インメモリで完結） |
| 適用場面 | 大規模ナレッジベースの Q&A | **単一〜少数ドキュメントの深い分析** |

**看護サマリー生成への適用:**

```
Map-Reduce (現行):
  全チャンク → 全情報を均一に抽出 → まとめる

Agentic Search (提案):
  テンプレート確認 → 必要な情報の検索計画を立案
    → LLM が検索ツールを使って医療記録を検索
    → 検索結果を評価、不足なら検索クエリを変えて再試行
    → 全セクションの情報が充足したら統合
```

### 2.3 エビデンス

| 出典 | 知見 |
|---|---|
| **A Picture of Agentic Search** (arXiv:2602.17518, 2026) | Agentic Search の定義論文。`<think>` → `<search>` → `<refine>` → `<answer>` のループを反復。人間の検索と比べ「実質的なクエリ書き換え・再送」パターンが多い。205,000+クエリの ASQ データセットを公開 |
| **APEX-Searcher** (arXiv:2603.13853, 2026) | Planning Agent + Execution Agent の2段階構成。HotpotQA で standard RAG 比 **+34.4%**（EM: 29.9% → 40.2%）。「複雑なクエリを分解し、段階的に情報を収集する」ことの有効性を実証 |
| **Morph LLM: Agentic Search** (2026) | コード検索での実装例。並列ツール呼び出し（4-12 同時検索）、サブエージェント分離によるコンテキスト汚染防止。「RAG は検索を前処理として扱うが、Agentic Search は検索をアクティブな推論プロセスとして扱う」 |
| **Chroma: Agentic Search Guide** (2026) | QueryPlanner → Executor → Evaluator の3コンポーネント構成。Plan → Search → Evaluate → Iterate の5段階ループ。「初期検索で不十分なら、エージェントがクエリを再構成して再検索する」|
| **Agentic Search Benchmark** (aimultiple.com, 2026) | Agentic Search の3層アーキテクチャ: Web retrieval → Orchestration → Reasoning/Generation。検索を自律的な推論プロセスとして扱うことで複雑なクエリへの対応力が向上 |
| **医療 RAG レビュー** (JMIR, 2025) | 医療ドメインで query decomposition + iterative retrieval により hallucination を低減。看護分野は研究の 6% と未開拓で差別化の余地大 |

### 2.4 看護サマリー生成における具体的改善

| 観点 | Map-Reduce | Agentic Search |
|---|---|---|
| 情報抽出の精度 | チャンク単位で均一抽出。テンプレートを意識しない | セクションごとに目的を持って検索。必要な数値を狙い撃ち |
| 情報の漏れ | 中間要約の圧縮過程で重要情報が脱落 | 情報不足を検知したら再検索で補完 |
| 文脈の一貫性 | チャンク境界で文脈が途切れる | 時系列を追って必要な期間の記録を横断検索 |
| テンプレート適合性 | Reduce 時に初めてテンプレートを考慮 | 最初からテンプレートのセクション構造を計画に反映 |
| 計算効率 | 全チャンクを処理（不要な情報も含む） | 必要な箇所のみ検索。不要な記録はスキップ |

---

## 3. システムアーキテクチャ全体図

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Client Layer                                                               │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────────┐                     │
│  │ client.py  │  │ 将来: Web UI │  │ 将来: DB Trigger  │                     │
│  │ (バッチ)    │  │             │  │                  │                     │
│  └─────┬──────┘  └──────┬──────┘  └────────┬─────────┘                     │
│        └────────────────┴──────────────────┘                               │
│                         │ HTTP POST /ask                                    │
└─────────────────────────┼───────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────────┐
│  API Layer (FastAPI)                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  POST /ask                                                           │   │
│  │  - リクエストバリデーション (Pydantic)                                  │   │
│  │  - InputAdapter でスキーマ正規化                                       │   │
│  │  - LangGraph グラフ実行                                               │   │
│  │  - レスポンス整形                                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────────┐
│  Orchestration Layer (LangGraph StateGraph)                                 │
│                                                                              │
│  ┌────────────┐                                                             │
│  │ input      │                                                             │
│  │ _adapter   │── テンプレートロード・チャンキング・検索インデックス構築        │
│  └─────┬──────┘                                                             │
│        │                                                                    │
│        ▼                                                                    │
│  ┌────────────────────────────────────────────────────────────────┐         │
│  │  Agentic Search Loop (テンプレートのセクションごとに反復)        │         │
│  │                                                                │         │
│  │  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐  │         │
│  │  │ plan     │──▶│ search   │──▶│ extract  │──▶│ evaluate │  │         │
│  │  │ (計画)   │   │ (検索)   │   │ (抽出)   │   │ (判定)   │  │         │
│  │  └──────────┘   └──────────┘   └──────────┘   └─────┬────┘  │         │
│  │       ▲                                              │       │         │
│  │       │          情報不足 → クエリ書き換えて再検索      │       │         │
│  │       └──────────────────────────────────────────────┘       │         │
│  └──────────────────────────────┬─────────────────────────────┘         │
│                                  │ 全セクション充足                       │
│                                  ▼                                       │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │  synthesize (統合生成)                                          │      │
│  │  抽出済みセクション情報 → テンプレートに沿ってドラフト生成        │      │
│  └──────────────────────────────┬─────────────────────────────────┘      │
│                                  │                                       │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │  Reflection Loop (上限付き早期終了)                              │      │
│  │  reflect ──▶ [APPROVED?] ──▶ revise ──▶ reflect (上限まで)     │      │
│  └──────────────────────────────┬─────────────────────────────────┘      │
│                                  │                                       │
│  ┌────────────────────────────────────────────────────────────────┐      │
│  │  output_formatter (テンプレートに基づく最終整形)                  │      │
│  └──────────────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────────────┐
│  LLM Layer (Ollama Python SDK)                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ollama.Client(host=OLLAMA_BASE_URL, timeout=600.0)                  │   │
│  │  model: gpt-oss:120b                                                 │   │
│  │  format: Pydantic model_json_schema() (構造化出力時)                  │   │
│  │  options: temperature=0.1, top_p=0.92, repeat_penalty=1.2            │   │
│  │  keep_alive: 60m                                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. State スキーマ設計

### 4.1 メイン State

```python
from typing import TypedDict, Annotated, Optional
import operator


class NursingSummaryState(TypedDict):
    """看護サマリー生成グラフのメイン状態。"""

    # --- 入力 ---
    patient_id: str                    # 患者ID
    raw_context: str                   # 生の医療記録テキスト
    hospital: str                      # 病院識別子 ("hanwa" | "shinkinen")

    # --- チャンキング・検索インデックス ---
    chunks: list[str]                  # 検索用に分割済みチャンクリスト
    chunk_index: list[dict]            # 各チャンクのメタデータ（日付範囲等）

    # --- テンプレート ---
    template_id: str                   # 使用するテンプレートID
    template: dict                     # ロード済みテンプレート定義

    # --- Agentic Search ---
    search_plan: list[dict]            # セクションごとの検索計画
    section_results: Annotated[dict[str, str], merge_dicts]
                                       # セクションkey → 抽出済み情報テキスト
    current_section_idx: int           # 現在処理中のセクションインデックス
    search_iteration: int              # 現セクションの検索反復回数
    max_search_iterations: int         # セクション当たりの検索上限（デフォルト 3）

    # --- 統合生成 ---
    draft_summary: str                 # 統合ドラフト

    # --- Reflection（上限付き早期終了） ---
    reflection_feedback: str           # LLM による改善指摘テキスト
    reflection_approved: bool          # LLM が「問題なし」と判断したか
    iteration_count: int               # 現在の反復回数
    max_iterations: int                # Reflection 上限回数（デフォルト 2）

    # --- 出力 ---
    final_summary: str                 # 最終看護サマリー
    error: Optional[str]               # エラーメッセージ


def merge_dicts(left: dict, right: dict) -> dict:
    """セクション結果を蓄積するカスタム reducer。"""
    merged = {**left}
    merged.update(right)
    return merged
```

### 4.2 検索計画の構造

```python
# search_plan の各要素
{
    "section_key": "nursing_process",
    "section_name": "入院中の看護の経過（生活状況）",
    "search_queries": [
        "入院時の状態とバイタルサイン",
        "看護問題と介入内容",
        "ADL変化とリハビリ経過",
    ],
    "required_info": [
        "バイタルサインの具体的数値",
        "日付ごとの経過",
        "看護介入の内容と結果",
    ],
}
```

---

## 5. LangGraph グラフ構造

### 5.1 グラフ定義

```
START
  │
  ▼
input_adapter ──── テンプレートロード・チャンク分割・検索インデックス構築
  │
  ▼
plan ──── テンプレートの各セクションに必要な情報と検索クエリを計画
  │
  ▼
search ──── 現セクションの検索クエリでチャンクを検索（キーワード + 意味検索）
  │
  ▼
extract ──── 検索結果からセクションに必要な情報を抽出
  │
  ▼
evaluate ──── conditional_edge
  │              │
  │ (充足 or     │ (情報不足 & search_iteration < max)
  │  上限到達)    ▼
  │           search (クエリを書き換えて再検索)
  │
  ▼
next_section ──── conditional_edge
  │              │
  │ (全セクション  │ (残りセクションあり)
  │  完了)        ▼
  │            plan (次セクションの検索開始)
  │
  ▼
synthesize ──── セクション別抽出結果をテンプレートに沿って統合・ドラフト生成
  │
  ▼
reflect ──── LLM テキスト批評
  │
  ▼
should_continue_reflection ──── conditional_edge
  │              │
  │ (完了)        │ (要改善 & iteration < max)
  │              ▼
  │           revise ──→ reflect
  │
  ▼
output_formatter
  │
  ▼
END
```

### 5.2 ノード一覧

| ノード名 | 責務 |
|---|---|
| `input_adapter` | テンプレートロード・テキスト分割・検索インデックス構築 |
| `plan` | テンプレートのセクション定義から検索計画を生成（LLM） |
| `search` | **LLM が検索ツールを自律的に選択・実行**（キーワード・日付・カテゴリ検索） |
| `extract` | 検索結果からセクションに必要な情報を LLM で抽出 |
| `evaluate` | 抽出結果の充足度を判定。不足ならクエリを書き換えて再検索 |
| `next_section` | 次のセクションに進むか、全完了かを判定 |
| `synthesize` | 全セクションの抽出結果をテンプレートに沿ってドラフト生成（LLM） |
| `reflect` | ドラフトの問題点をテキストで指摘（LLM） |
| `revise` | フィードバックに基づくドラフト改善（LLM） |
| `output_formatter` | テンプレート書式で最終整形 |

### 5.3 条件付きエッジ

| エッジ関数 | 起点 | 分岐先 | 条件 |
|---|---|---|---|
| `evaluate_sufficiency` | `evaluate` | `search` / `next_section` | 情報不足 & 検索上限未到達 → 再検索 |
| `has_more_sections` | `next_section` | `plan` / `synthesize` | 未処理セクションあり → 次セクション |
| `should_continue_reflection` | `reflect` | `revise` / `output_formatter` | `!approved and iteration < max` |

---

## 6. 各ノードの詳細設計

### 6.1 input_adapter

```python
def input_adapter(state: NursingSummaryState) -> dict:
    """テンプレートロード・テキスト分割・検索インデックス構築。

    医療記録を意味的に検索可能な単位に分割し、
    各チャンクに日付等のメタデータを付与する。
    Map-Reduce と異なり、チャンクは「検索対象」として使い、
    全チャンクを均一に処理するのではなく必要な箇所のみを検索する。
    """
    # テンプレートロード
    template = load_template(state["template_id"])

    # 検索用にセマンティックチャンクへ分割（日付単位）
    chunks, chunk_index = build_search_index(state["raw_context"])

    return {
        "template": template,
        "chunks": chunks,
        "chunk_index": chunk_index,
        "current_section_idx": 0,
        "search_iteration": 0,
        "section_results": {},
    }


def build_search_index(text: str) -> tuple[list[str], list[dict]]:
    """医療記録を日付単位のチャンクに分割し検索インデックスを構築する。

    固定トークン数ではなく、日付（YYYYMMDD）パターンを境界として
    臨床的に意味のある単位で分割する。

    Returns:
        (チャンクテキストのリスト, メタデータのリスト)
    """
    # 日付パターン（"- 20230209" 等）で分割
    # フォールバック: 日付が見つからない場合は固定トークン数で分割
    ...
```

### 6.2 plan

```python
PLAN_PROMPT = """あなたは看護サマリー作成の専門家です。
以下のテンプレートセクションに必要な情報を収集するための検索計画を立ててください。

## 対象セクション
名前: {section_name}
説明: {section_description}

## 医療記録の日付範囲
{available_dates}

## 指示
このセクションを完成させるために、医療記録のどの部分を検索すべきか、
具体的な検索キーワードを3〜5個生成してください。
検索キーワードは医療記録に含まれそうな具体的な用語を使ってください。
"""


def plan(state: NursingSummaryState) -> dict:
    """現セクションに必要な情報の検索計画を LLM で生成する。

    テンプレートのセクション定義を元に、医療記録から
    どのような情報を検索すべきかの計画を立てる。
    """
    section = state["search_plan"][state["current_section_idx"]]
    dates = [c.get("date", "不明") for c in state["chunk_index"]]

    prompt = PLAN_PROMPT.format(
        section_name=section["section_name"],
        section_description=section.get("description", ""),
        available_dates=", ".join(sorted(set(dates))),
    )

    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        format=SearchQueries.model_json_schema(),
        options={**LLM_OPTIONS, "temperature": 0.0},
        keep_alive="60m",
    )

    queries = SearchQueries.model_validate_json(response.message.content)
    plan = state["search_plan"].copy()
    plan[state["current_section_idx"]]["search_queries"] = queries.queries

    return {
        "search_plan": plan,
        "search_iteration": 0,
    }


class SearchQueries(BaseModel):
    """検索クエリの構造化出力。"""
    queries: list[str]
```

### 6.3 search

Agentic Search の核心: **LLM が検索ツールを自律的に選択・実行する**。
ベクトル DB は使わず、以下の検索ツールを LLM に提供する。

```python
# --- 検索ツール定義（LLM が呼び出し可能） ---

def search_by_keyword(keyword: str) -> list[str]:
    """キーワードで医療記録チャンクを検索する。

    Args:
        keyword: 検索キーワード（例: "バイタル", "SpO2", "セフトリアキソン"）。

    Returns:
        キーワードを含むチャンクのリスト。
    """
    ...


def search_by_date_range(start_date: str, end_date: str) -> list[str]:
    """日付範囲で医療記録チャンクを検索する。

    Args:
        start_date: 開始日（YYYYMMDD 形式）。
        end_date: 終了日（YYYYMMDD 形式）。

    Returns:
        指定期間のチャンクリスト。
    """
    ...


def search_by_category(category: str) -> list[str]:
    """カテゴリ（カルテ種別等）でチャンクを検索する。"""
    ...


# --- search ノード ---

def search(state: NursingSummaryState) -> dict:
    """LLM が検索ツールを自律的に選択・実行する。

    Agentic Search の核心: LLM が検索戦略を推論し、
    適切なツールを呼び出して必要な情報を収集する。
    ベクトル DB は使わず、キーワード・日付・カテゴリの
    検索ツールを LLM に提供する。
    """
    section = state["search_plan"][state["current_section_idx"]]

    # LLM にツール選択を委譲（Ollama の tool calling を使用）
    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=[{
            "role": "user",
            "content": f"セクション「{section['section_name']}」に必要な情報を "
                       f"検索ツールを使って医療記録から探してください。"
                       f"\n日付範囲: {get_date_range(state['chunk_index'])}"
                       f"\nチャンク数: {len(state['chunks'])}",
        }],
        tools=[search_by_keyword, search_by_date_range, search_by_category],
        keep_alive="60m",
    )

    # ツール呼び出し結果を収集
    results = []
    if response.message.tool_calls:
        for tool_call in response.message.tool_calls:
            fn = tool_call.function
            result = execute_search_tool(
                fn.name, fn.arguments,
                state["chunks"], state["chunk_index"],
            )
            results.extend(result)

    return {"_search_results": list(dict.fromkeys(results))}  # 重複除去
```

### 6.4 extract

```python
EXTRACT_PROMPT = """以下の医療記録の抜粋から、指定されたセクションに必要な情報を抽出してください。

## 対象セクション
名前: {section_name}
説明: {section_description}

## 医療記録の抜粋
{search_results}

## 指示
- 具体的な数値（バイタルサイン、投薬量等）を必ず含めてください
- 日付を YYYY年MM月DD日 形式で記載してください
- 記録にない情報は「記録なし」と明記してください
- 推測や憶測は含めないでください
"""


def extract(state: NursingSummaryState) -> dict:
    """検索結果からセクションに必要な情報を LLM で抽出する。

    search ノードが返したチャンクを LLM に渡し、
    セクション定義に基づいて必要な情報を抽出する。
    """
    section = state["search_plan"][state["current_section_idx"]]
    search_results = state.get("_search_results", [])

    prompt = EXTRACT_PROMPT.format(
        section_name=section["section_name"],
        section_description=section.get("description", ""),
        search_results="\n\n---\n\n".join(search_results),
    )

    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options=LLM_OPTIONS,
        keep_alive="60m",
    )

    # セクション結果を蓄積（merge_dicts reducer で統合）
    section_key = section["section_key"]
    existing = state["section_results"].get(section_key, "")
    updated = existing + "\n" + response.message.content if existing else response.message.content

    return {
        "section_results": {section_key: updated},
        "search_iteration": state["search_iteration"] + 1,
    }
```

### 6.5 evaluate（条件付きエッジ関数）

```python
EVALUATE_PROMPT = """以下はあるセクションに対して抽出された情報です。
このセクションの要件を満たすのに十分な情報が揃っていますか？

## セクション: {section_name}
## 要件: {section_description}

## 抽出済み情報
{extracted_info}

## 指示
情報が十分であれば「SUFFICIENT」とだけ回答してください。
不足している場合は、追加で検索すべき具体的なキーワードを箇条書きで回答してください。
"""


def evaluate(state: NursingSummaryState) -> dict:
    """抽出結果の充足度を LLM で判定し、不足時は追加クエリを生成する。"""
    section = state["search_plan"][state["current_section_idx"]]
    section_key = section["section_key"]
    extracted = state["section_results"].get(section_key, "")

    prompt = EVALUATE_PROMPT.format(
        section_name=section["section_name"],
        section_description=section.get("description", ""),
        extracted_info=extracted,
    )

    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={**LLM_OPTIONS, "temperature": 0.0},
        keep_alive="60m",
    )

    feedback = response.message.content
    is_sufficient = "SUFFICIENT" in feedback.upper()

    if not is_sufficient:
        # 追加検索クエリで search_plan を更新
        plan = state["search_plan"].copy()
        new_queries = [
            line.strip("- ").strip()
            for line in feedback.split("\n")
            if line.strip().startswith("-")
        ]
        plan[state["current_section_idx"]]["search_queries"] = new_queries
        return {"search_plan": plan, "_section_sufficient": False}

    return {"_section_sufficient": True}


def evaluate_sufficiency(
    state: NursingSummaryState,
) -> Literal["search", "next_section"]:
    """検索を続行するか次のセクションに進むかを判定する。"""
    if state.get("_section_sufficient", False):
        return "next_section"
    if state["search_iteration"] >= state["max_search_iterations"]:
        return "next_section"
    return "search"
```

### 6.6 next_section

```python
def next_section(state: NursingSummaryState) -> dict:
    """次のセクションに進む。"""
    return {"current_section_idx": state["current_section_idx"] + 1}


def has_more_sections(
    state: NursingSummaryState,
) -> Literal["plan", "synthesize"]:
    """未処理セクションが残っているかを判定する。"""
    if state["current_section_idx"] < len(state["search_plan"]):
        return "plan"
    return "synthesize"
```

### 6.7 synthesize

```python
SYNTHESIZE_PROMPT = """以下のセクション別に抽出された情報を統合し、
テンプレートに沿った看護サマリーのドラフトを作成してください。

## テンプレート
{format_instruction}

## セクション別抽出情報
{section_data}

## 指示
- 各セクションの情報を統合し、時系列順に整理してください
- バイタルサイン・投薬量等の具体的数値を必ず含めてください
- 情報がないセクションは「記録なし」としてください
- 推測は含めず、抽出された事実のみを記載してください
"""


def synthesize(state: NursingSummaryState) -> dict:
    """全セクションの抽出結果をテンプレートに沿って統合する。"""
    format_instruction = build_format_instruction(state["template"])

    section_data = "\n\n".join(
        f"### {key}\n{value}"
        for key, value in state["section_results"].items()
    )

    prompt = SYNTHESIZE_PROMPT.format(
        format_instruction=format_instruction,
        section_data=section_data,
    )

    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options=LLM_OPTIONS,
        keep_alive="60m",
    )

    return {"draft_summary": response.message.content}
```

### 6.8 reflect

```python
REFLECTION_PROMPT = """あなたは看護サマリーの品質管理担当者です。
以下の看護サマリーを批評し、具体的な問題点があれば指摘してください。

## 確認観点
- バイタルサイン・投薬量に具体的な数値が記載されているか
- 日付が時系列順に整理されているか
- 指定フォーマットの全セクションが埋まっているか
- 推測・憶測が混入していないか（客観的事実のみか）
- 医療略語が初出時に正式名称で展開されているか

## 対象テンプレートのセクション
{template_sections}

## 看護サマリー
{draft_summary}

## 出力形式
問題がある場合:
改善すべき具体的な箇所を箇条書きで列挙してください。
各指摘には「どのセクションの」「何が」「どう問題か」を明記してください。

問題がない場合:
「APPROVED」とだけ回答してください。
"""


def reflect(state: NursingSummaryState) -> dict:
    """ドラフトサマリーの具体的な問題点をテキストで指摘する。

    LLM にスコアリングは求めず、テキストベースで問題点を列挙させる。
    問題がなければ LLM は "APPROVED" を返し、早期終了のトリガーとなる。

    Args:
        state: draft_summary と template を含むグラフ状態。

    Returns:
        reflection_feedback, reflection_approved, iteration_count を含む更新辞書。
    """
    sections = state["template"].get("sections", [])
    section_names = "\n".join(
        f"- {s['name']}" for s in sections
    )

    prompt = REFLECTION_PROMPT.format(
        template_sections=section_names,
        draft_summary=state["draft_summary"],
    )

    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options={**LLM_OPTIONS, "temperature": 0.0},
        keep_alive="60m",
    )

    feedback = response.message.content
    approved = "APPROVED" in feedback.upper()

    return {
        "reflection_feedback": feedback,
        "reflection_approved": approved,
        "iteration_count": state["iteration_count"] + 1,
    }
```

### 6.9 revise

```python
REVISION_PROMPT = """以下の看護サマリーを、指摘された問題点に基づいて改善してください。

## 元のサマリー
{draft_summary}

## 指摘された問題点
{feedback}

## 指示
- 指摘された箇所のみを修正してください
- 正しい部分は変更しないでください
- 修正後のサマリー全体を出力してください
"""


def revise(state: NursingSummaryState) -> dict:
    """フィードバックに基づいてドラフトを改善する。

    reflect ノードが指摘した具体的な問題箇所のみを修正する。

    Args:
        state: draft_summary と reflection_feedback を含むグラフ状態。

    Returns:
        更新された draft_summary を含む辞書。
    """
    prompt = REVISION_PROMPT.format(
        draft_summary=state["draft_summary"],
        feedback=state["reflection_feedback"],
    )

    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        options=LLM_OPTIONS,
        keep_alive="60m",
    )

    return {"draft_summary": response.message.content}
```

### 6.10 should_continue_reflection（条件付きエッジ関数）

```python
from typing import Literal


def should_continue_reflection(
    state: NursingSummaryState,
) -> Literal["revise", "output_formatter"]:
    """Reflection ループの継続を判定する。

    LLM が「問題なし」(APPROVED) と判断した場合は早期終了。
    問題ありでも上限回数に達した場合は終了する。
    LLM スコアやルールベース判定は使用しない。

    Args:
        state: reflection_approved, iteration_count, max_iterations を含むグラフ状態。

    Returns:
        "revise"（改善継続）または "output_formatter"（出力へ）。
    """
    # LLM が問題なしと判断 → 早期終了
    if state["reflection_approved"]:
        return "output_formatter"

    # 上限到達 → 終了
    if state["iteration_count"] >= state["max_iterations"]:
        return "output_formatter"

    # 問題あり & 上限未到達 → 改善継続
    return "revise"
```

### 6.11 output_formatter

```python
def output_formatter(state: NursingSummaryState) -> dict:
    """最終的な看護サマリーを設定する。

    Reflection を通過したドラフトを最終サマリーとして確定する。

    Args:
        state: draft_summary を含むグラフ状態。

    Returns:
        final_summary を含む更新辞書。
    """
    return {"final_summary": state["draft_summary"]}
```

---

## 7. グラフ構築コード

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy


def build_nursing_summary_graph() -> StateGraph:
    """看護サマリー生成グラフを構築してコンパイルする。"""
    builder = StateGraph(NursingSummaryState)

    # --- ノード登録 ---
    builder.add_node("input_adapter", input_adapter)
    builder.add_node("plan", plan, retry_policy=RetryPolicy(max_attempts=2))
    builder.add_node("search", search)  # LLM 不使用 → リトライ不要
    builder.add_node("extract", extract, retry_policy=RetryPolicy(max_attempts=2))
    builder.add_node("evaluate", evaluate)
    builder.add_node("next_section", next_section)
    builder.add_node("synthesize", synthesize, retry_policy=RetryPolicy(max_attempts=2))
    builder.add_node("reflect", reflect)
    builder.add_node("revise", revise)
    builder.add_node("output_formatter", output_formatter)

    # --- エッジ定義 ---

    # 入力 → 計画
    builder.add_edge(START, "input_adapter")
    builder.add_edge("input_adapter", "plan")

    # Agentic Search ループ: plan → search → extract → evaluate
    builder.add_edge("plan", "search")
    builder.add_edge("search", "extract")
    builder.add_edge("extract", "evaluate")

    # evaluate → 再検索 or 次セクション
    builder.add_conditional_edges(
        "evaluate",
        evaluate_sufficiency,
        {"search": "search", "next_section": "next_section"},
    )

    # next_section → 次セクション or 統合
    builder.add_conditional_edges(
        "next_section",
        has_more_sections,
        {"plan": "plan", "synthesize": "synthesize"},
    )

    # 統合 → Reflection
    builder.add_edge("synthesize", "reflect")

    # Reflection ループ
    builder.add_conditional_edges(
        "reflect",
        should_continue_reflection,
        {"revise": "revise", "output_formatter": "output_formatter"},
    )
    builder.add_edge("revise", "reflect")

    # 出力
    builder.add_edge("output_formatter", END)

    return builder.compile()
```

---

## 8. Ollama クライアント設定

```python
import httpx
from ollama import Client

# --- 定数 ---
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "130000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "1000"))

# --- Ollama クライアント ---
ollama_client = Client(
    host=OLLAMA_BASE_URL,
    timeout=httpx.Timeout(
        connect=30.0,   # 接続確立: 30秒
        read=600.0,     # 読み取り: 10分（120B モデルの生成待ち）
        write=30.0,     # 書き込み: 30秒
        pool=30.0,      # プール取得: 30秒
    ),
)

# --- LLM 共通オプション ---
LLM_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.92,
    "repeat_penalty": 1.2,
    "num_ctx": 8192,
    "num_predict": 4096,
}
```

---

## 9. FastAPI 統合

Flask から FastAPI への移行により以下の利点を得る:

| 項目 | Flask | FastAPI |
|---|---|---|
| 非同期処理 | 非対応（Gunicorn + gevent 等が必要） | **ネイティブ async/await** |
| リクエストバリデーション | 手動 | **Pydantic モデルで自動バリデーション** |
| API ドキュメント | なし（Swagger 別途導入） | **自動生成（/docs に Swagger UI）** |
| 型安全性 | 弱い | **リクエスト/レスポンス共に型定義** |
| パフォーマンス | WSGI（同期） | **ASGI（非同期・高スループット）** |

```python
import os
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# --- リクエスト/レスポンスモデル ---

class AskRequest(BaseModel):
    """看護サマリー生成リクエスト。"""

    context: str = Field(..., min_length=1, description="医療記録テキスト")
    patient_id: str = Field(default="unknown", description="患者ID")
    template_id: str | None = Field(
        default=None,
        description="テンプレートID（省略時は HOSPITAL 環境変数）",
    )


class AskResponse(BaseModel):
    """看護サマリー生成レスポンス。"""

    answer: str = Field(description="生成された看護サマリー")
    template_id: str = Field(description="使用されたテンプレートID")
    iteration_count: int = Field(description="Reflection の反復回数")


class TemplateInfo(BaseModel):
    """テンプレート情報。"""

    id: str
    name: str
    description: str = ""


class ErrorResponse(BaseModel):
    """エラーレスポンス。"""

    error: str
    trace: str | None = None


# --- アプリケーション ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時にグラフを構築する。"""
    app.state.graph = build_nursing_summary_graph()
    yield


app = FastAPI(
    title="CareSummaryGen API",
    description="医療記録から看護サマリーを自動生成する API",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def index() -> dict:
    """ヘルスチェック。"""
    return {
        "status": "ok",
        "model": MODEL_NAME,
    }


@app.get("/templates", response_model=list[TemplateInfo])
async def get_templates() -> list[TemplateInfo]:
    """利用可能なテンプレート一覧を返す。"""
    return [TemplateInfo(**t) for t in list_templates()]


@app.post(
    "/ask",
    response_model=AskResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
async def ask(req: AskRequest) -> AskResponse:
    """看護サマリーを生成する。

    医療記録テキストを受け取り、指定テンプレートに沿って
    Agentic Search + Reflection で看護サマリーを生成する。
    """
    hospital = os.environ.get("HOSPITAL", "hanwa")
    template_id = req.template_id or hospital

    try:
        template = load_template(template_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    initial_state = {
        "patient_id": req.patient_id,
        "raw_context": req.context,
        "hospital": hospital,
        "chunks": [],
        "chunk_index": [],
        "template_id": template_id,
        "template": template,
        "search_plan": [],
        "section_results": {},
        "current_section_idx": 0,
        "search_iteration": 0,
        "max_search_iterations": int(
            os.environ.get("MAX_SEARCH_ITERATIONS", "3")
        ),
        "draft_summary": "",
        "reflection_feedback": "",
        "reflection_approved": False,
        "iteration_count": 0,
        "max_iterations": int(
            os.environ.get("MAX_REFLECTION_ITERATIONS", "2")
        ),
        "final_summary": "",
        "error": None,
    }

    try:
        result = app.state.graph.invoke(initial_state)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        return AskResponse(
            answer=result["final_summary"],
            template_id=result["template_id"],
            iteration_count=result["iteration_count"],
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"エラーが発生しました: {e}",
        )
```

### 起動コマンド

```bash
# 開発
uv run uvicorn app:app --reload --port 5000

# 本番
uv run uvicorn app:app --host 0.0.0.0 --port 5000 --workers 1
# workers=1: gpt-oss:120b は同時1リクエストが推奨のため
```

---

## 9. 入力スキーマ柔軟性（InputAdapter パターン）

将来のデータベース連携に備えたアダプター設計。

```python
from abc import ABC, abstractmethod


class BaseInputAdapter(ABC):
    """入力データを統一的な Markdown テキストに変換する基底クラス。"""

    @abstractmethod
    def to_markdown(self, raw_data: Any) -> str:
        """任意の入力データを Markdown テキストに変換する。"""
        ...


class PlainTextAdapter(BaseInputAdapter):
    """プレーンテキスト / Markdown 入力用アダプター（現行互換）。"""

    def to_markdown(self, raw_data: str) -> str:
        return raw_data


class JsonRecordAdapter(BaseInputAdapter):
    """JSON 形式の医療記録用アダプター（将来の DB 連携用）。"""

    def to_markdown(self, raw_data: dict) -> str:
        lines = [f"# 患者ID: {raw_data.get('patient_id', 'unknown')}"]
        for record in raw_data.get("records", []):
            date = record.get("date", "不明")
            content = record.get("content", "")
            lines.append(f"\n- {date}")
            lines.append(f"  - カルテ")
            for line in content.split("\n"):
                lines.append(f"    {line}")
        return "\n".join(lines)


# アダプターレジストリ
ADAPTERS: dict[str, type[BaseInputAdapter]] = {
    "text": PlainTextAdapter,
    "json": JsonRecordAdapter,
}


def get_adapter(schema_type: str = "text") -> BaseInputAdapter:
    """スキーマタイプに応じたアダプターを返す。"""
    adapter_cls = ADAPTERS.get(schema_type, PlainTextAdapter)
    return adapter_cls()
```

---

## 10. テンプレート管理システム

機関ごとに異なる看護サマリーの体裁を、外部テンプレートファイルで管理する。

### 10.1 テンプレートファイル形式（YAML）

```yaml
# templates/hanwa.yaml
id: hanwa
name: 阪和病院
description: 阪和病院の看護サマリーフォーマット

sections:
  - name: 指導した内容
    key: instruction
    description: 患者・家族に対して指導した内容
    required: true

  - name: 医療機器装着・挿入・処置部位
    key: medical_equipment
    description: 装着中の医療機器、挿入物、処置部位の情報
    required: true

  - name: 入院中の看護の経過（生活状況）
    key: nursing_process
    description: 入院中の看護経過と生活状況の変化
    required: true

  - name: 患者への病状説明及び本人・家族の受け止め方
    key: patient_condition
    description: 病状説明の内容と患者・家族の反応
    required: true

  - name: 継続される問題（今後のリスク）
    key: risks
    description: 退院後も継続する問題点と今後のリスク
    required: true

  - name: その他
    key: others
    description: 上記に該当しない重要事項
    required: false

# セクション区切りの書式
section_delimiter: "--- {name} ---"

# Few-Shot 例示ファイル（オプション）
examples:
  - input: examples/hanwa/input_01.md
    output: examples/hanwa/output_01.md
  - input: examples/hanwa/input_02.md
    output: examples/hanwa/output_02.md
```

```yaml
# templates/shinkinen.yaml
id: shinkinen
name: 新記念病院
description: 新記念病院の看護サマリーフォーマット

sections:
  - name: 入院中の経過及び看護上の問題経過
    key: progress
    description: 日付ごとの入院経過と看護上の問題
    required: true

  - name: 備考
    key: remarks
    description: 退院後の注意事項、外来通院、サービス情報
    required: true

section_delimiter: "--- {name} ---"

examples:
  - input: examples/shinkinen/input_01.md
    output: examples/shinkinen/output_01.md
```

### 10.2 テンプレートローダー

```python
import yaml
from pathlib import Path

TEMPLATE_DIR = Path("templates")


def load_template(template_id: str) -> dict:
    """テンプレートIDに対応するYAMLファイルをロードする。

    Args:
        template_id: テンプレート識別子（例: "hanwa", "shinkinen"）。

    Returns:
        テンプレート定義の辞書。

    Raises:
        FileNotFoundError: テンプレートファイルが存在しない場合。
    """
    path = TEMPLATE_DIR / f"{template_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"テンプレート '{template_id}' が見つかりません: {path}"
        )

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def list_templates() -> list[dict]:
    """利用可能なテンプレートの一覧を返す。"""
    templates = []
    for path in TEMPLATE_DIR.glob("*.yaml"):
        with open(path, encoding="utf-8") as f:
            t = yaml.safe_load(f)
            templates.append({
                "id": t["id"],
                "name": t["name"],
                "description": t.get("description", ""),
            })
    return templates


def build_format_instruction(template: dict) -> str:
    """テンプレートからLLMへのフォーマット指示文を構築する。

    テンプレートの sections 定義を元に、
    Reduce プロンプトに埋め込むフォーマット指示を生成する。

    Args:
        template: ロード済みテンプレート辞書。

    Returns:
        フォーマット指示文字列。
    """
    delimiter = template.get("section_delimiter", "--- {name} ---")
    lines = ["以下のフォーマットに従って出力してください:\n"]

    for section in template["sections"]:
        header = delimiter.format(name=section["name"])
        lines.append(header)
        lines.append(f"[{section['description']}]")
        lines.append("")

    return "\n".join(lines)
```

### 10.3 テンプレートの使用箇所

| ノード | テンプレートの使い方 |
|---|---|
| `input_adapter` | `template_id` から YAML をロードして state に格納 |
| `reduce` / `direct_reduce` | `build_format_instruction(template)` で Reduce プロンプトにフォーマット指示を埋め込む |
| `reflect` | テンプレートの `sections` 一覧を参照し、全セクションの充足を確認観点に含める |
| `output_formatter` | テンプレートの `section_delimiter` に基づいて最終整形 |

### 10.4 API でのテンプレート指定

```
POST /ask
{
    "context": "<医療記録テキスト>",
    "template_id": "hanwa"        # オプション。省略時は HOSPITAL 環境変数
}

GET /templates
→ [{"id": "hanwa", "name": "阪和病院", ...}, {"id": "shinkinen", ...}]
```

### 10.5 テンプレート追加の手順

新しい機関のフォーマットを追加する場合:

1. `templates/<機関ID>.yaml` を作成（上記フォーマットに従う）
2. `examples/<機関ID>/` に Few-Shot 用の入出力例を配置（オプション）
3. API リクエストで `"template_id": "<機関ID>"` を指定

コード変更は不要。YAML 追加のみで新フォーマットに対応できる。

---

## 11. エラーハンドリング戦略

### 10.1 レイヤー別対策

| レイヤー | エラー種別 | 対策 |
|---|---|---|
| Ollama 接続 | `ConnectionError` | LangGraph `RetryPolicy`（3回、指数バックオフ） |
| Ollama 生成 | `ResponseError` (5xx) | LangGraph `RetryPolicy` |
| Ollama OOM | `ResponseError` | エラー state に記録、グラフ終了 |
| 構造化出力パース | `ValidationError` | try/except でフォールバック（テキスト出力） |
| タイムアウト | `httpx.TimeoutException` | `RetryPolicy` + 長めの read timeout |

### 10.2 Reflection のフォールバック

```python
def reflect(state: NursingSummaryState) -> dict:
    """品質批評。LLM 呼び出し失敗時はそのまま承認にフォールバック。"""
    try:
        # ... 通常の reflect 処理 ...
        return {
            "reflection_feedback": feedback,
            "reflection_approved": "APPROVED" in feedback.upper(),
            "iteration_count": state["iteration_count"] + 1,
        }
    except Exception:
        # LLM 呼び出しが失敗した場合はドラフトをそのまま承認
        return {
            "reflection_feedback": "",
            "reflection_approved": True,
            "iteration_count": state["iteration_count"] + 1,
        }
```

---

## 12. ディレクトリ構造（移行後）

```
CareSummaryGen/
├── app.py                          # FastAPI エントリーポイント
├── client.py                       # バッチクライアント（変更最小限）
├── pyproject.toml                  # 依存管理 (uv)
├── .env                            # 環境変数
│
├── templates/                      # 機関別出力テンプレート (YAML)
│   ├── hanwa.yaml                  # 阪和病院
│   ├── shinkinen.yaml              # 新記念病院
│   └── ...                         # 新規機関は YAML 追加のみ
│
├── examples/                       # Few-Shot 例示データ（テンプレートから参照）
│   ├── hanwa/
│   │   ├── input_01.md
│   │   └── output_01.md
│   └── shinkinen/
│       ├── input_01.md
│       └── output_01.md
│
├── graph/                          # LangGraph グラフ定義
│   ├── __init__.py
│   ├── state.py                    # State TypedDict 定義
│   ├── builder.py                  # グラフ構築・コンパイル
│   ├── nodes/                      # ノード関数
│   │   ├── __init__.py
│   │   ├── input_adapter.py        # 入力正規化・検索インデックス構築
│   │   ├── plan.py                 # 検索計画立案
│   │   ├── search.py               # 検索ツール定義・LLM 検索実行
│   │   ├── extract.py              # 検索結果からの情報抽出
│   │   ├── evaluate.py             # 情報充足度判定
│   │   ├── synthesize.py           # セクション別情報の統合ドラフト生成
│   │   ├── reflect.py              # テキスト批評
│   │   ├── revise.py               # ドラフト改善
│   │   └── output_formatter.py     # テンプレート整形
│   └── edges/                      # 条件付きエッジ関数
│       ├── __init__.py
│       ├── search_loop.py          # evaluate_sufficiency, has_more_sections
│       └── reflection.py           # should_continue_reflection
│
├── llm/                            # LLM クライアント
│   ├── __init__.py
│   ├── client.py                   # Ollama Client ラッパー
│   └── prompts.py                  # プロンプトテンプレート
│
├── adapters/                       # 入力スキーマアダプター
│   ├── __init__.py
│   ├── base.py                     # BaseInputAdapter
│   ├── text.py                     # PlainTextAdapter
│   └── json_record.py             # JsonRecordAdapter（将来用）
│
├── templates_loader/                # テンプレート管理
│   ├── __init__.py
│   ├── loader.py                   # YAML ロード・一覧・フォーマット指示構築
│   └── models.py                   # テンプレートの Pydantic モデル
│
├── config/                         # 設定管理
│   ├── __init__.py
│   └── settings.py                 # 環境変数・定数
│
├── utils/                          # ユーティリティ（既存）
│   ├── preprocess.py
│   ├── extract_data.py
│   ├── data_collection.py
│   └── preprocess_hanwa_kinen_1.py
│
├── data/                           # テストデータ
│   ├── test_sample1.md
│   └── test_sample2.md
│
└── docs/                           # ドキュメント
    └── architecture.md             # 本設計書
```

---

## 13. 依存パッケージ（移行後）

```toml
[project]
dependencies = [
    # --- オーケストレーション ---
    "langgraph>=1.1.0",          # StateGraph, Send, RetryPolicy
    # langchain-core は langgraph の依存として自動インストール

    # --- LLM ---
    "ollama>=0.6.0",             # Ollama Python SDK

    # --- Web ---
    "fastapi>=0.115.0",          # ASGI Web フレームワーク
    "uvicorn[standard]>=0.34.0", # ASGI サーバー

    # --- データ処理 ---
    "tiktoken>=0.8.0",           # トークン計算
    "pydantic>=2.7.4",           # 構造化出力・バリデーション
    "pyyaml>=6.0",               # テンプレート YAML ロード

    # --- 前処理（既存） ---
    "chardet>=5.0.0",            # エンコーディング検出
    "beautifulsoup4>=4.12.0",    # HTML タグ除去
]

# 以下は削除（LangChain 関連）
# "langchain>=0.3.7"
# "langchain-community>=0.3.7"
# "langchain-chroma>=0.1.4"
# "transformers>=4.46.3"
# "sentence-transformers>=3.3.1"
# "torch>=2.5.1"
# "chromadb>=0.5.20"
# "unstructured>=0.16.6"
```

**削除される依存の影響**:
- `torch`, `transformers`, `sentence-transformers`: RAG 廃止に伴い不要
- `chromadb`, `langchain-chroma`: ベクトル DB 不使用
- `unstructured`: 前処理で未使用

---

## 14. 環境変数

```bash
# .env
OLLAMA_MODEL=gpt-oss:120b
OLLAMA_BASE_URL=http://localhost:11434
HOSPITAL=hanwa                        # デフォルトのテンプレートID
CHUNK_SIZE=130000
CHUNK_OVERLAP=1000
MAX_REFLECTION_ITERATIONS=2           # Reflection ループの上限回数
HF_TOKEN=<必要に応じて>
```

---

## 15. データフロー（移行後）

```
client.py → POST /ask {"context": "医療記録テキスト", "template_id": "hanwa"}
    │
    ▼
FastAPI app.py → graph.invoke(initial_state)
    │
    ├─ input_adapter: テンプレートロード + 日付単位チャンク分割 + 検索インデックス構築
    │
    ├─ === Agentic Search ループ (セクションごとに反復) ===
    │   │
    │   ├─ plan: テンプレートのセクション定義から検索計画を LLM で生成
    │   │
    │   ├─ search: LLM が検索ツール (keyword/date/category) を自律的に呼び出し
    │   │
    │   ├─ extract: 検索結果からセクションに必要な情報を LLM で抽出
    │   │
    │   ├─ evaluate: 情報充足度を LLM で判定
    │   │   ├─ [不足 & 上限未到達] → search (クエリ書き換えて再検索)
    │   │   └─ [充足 or 上限到達] → next_section
    │   │
    │   └─ next_section:
    │       ├─ [残りセクションあり] → plan (次セクション)
    │       └─ [全セクション完了] → synthesize
    │
    ├─ synthesize: セクション別抽出結果をテンプレートに沿って統合ドラフト生成
    │
    ├─ reflect: テキスト批評（問題点の列挙 or "APPROVED"）
    │   ├─ [APPROVED or iteration >= max] → output_formatter
    │   └─ [問題あり] → revise → reflect (上限付きループ)
    │
    └─ output_formatter → テンプレートに基づく最終整形 → final_summary
        │
        ▼
FastAPI response: {"answer": final_summary, "template_id": "hanwa"}
```

---

## 16. 移行計画

### Phase 1: 基盤構築（1-2週間）
- [ ] `graph/state.py` — State TypedDict 定義
- [ ] `llm/client.py` — Ollama Client ラッパー
- [ ] `llm/prompts.py` — 現行プロンプトの移植
- [ ] `config/settings.py` — 環境変数・定数の集約
- [ ] `templates/hanwa.yaml` — 阪和病院テンプレート作成
- [ ] `templates/shinkinen.yaml` — 新記念病院テンプレート作成
- [ ] `templates_loader/loader.py` — テンプレートローダー実装
- [ ] `pyproject.toml` — 依存パッケージの更新

### Phase 2: グラフ実装（1-2週間）
- [ ] `graph/nodes/input_adapter.py`
- [ ] `graph/nodes/map_worker.py`
- [ ] `graph/nodes/reduce.py` + `direct_reduce`（テンプレート統合）
- [ ] `graph/edges/routing.py` — Send API 実装
- [ ] `graph/builder.py` — グラフ構築・コンパイル
- [ ] 単体テスト: 各ノードの入出力検証

### Phase 3: Reflection 追加（1週間）
- [ ] `graph/nodes/reflect.py` — テキスト批評 + APPROVED 判定
- [ ] `graph/nodes/revise.py`
- [ ] `graph/edges/reflection.py` — should_continue_reflection
- [ ] 上限付き早期終了の動作検証

### Phase 4: FastAPI 統合 & テスト（1週間）
- [ ] `app.py` — FastAPI + LangGraph グラフ統合 + テンプレート API
- [ ] `client.py` — 動作確認（変更最小限）
- [ ] test_sample1.md / test_sample2.md での E2E テスト
- [ ] 現行出力との比較検証

### Phase 5: クリーンアップ
- [ ] 不要な依存の削除
- [ ] old/ ディレクトリの整理
- [ ] ドキュメント更新
