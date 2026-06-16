# DBクエリ前提の入力取得・整理・LLM受け渡し 詳細設計

本書は、看護サマリ生成の入力をデータベースからクエリで取得する場合の取得・整理・LLM受け渡しの詳細設計を定義する。
取得するカラムは未確定であることを前提に、カラム確定なしでも動作する構成とする。

本書は、4観点のWeb調査（看護サマリ作成プロセス／電子カルテ・看護記録のデータ構造／医療情報標準／LLM前処理）と、
3観点の設計レビュー（完全性・柔軟性／既存パイプライン統合・実装可能性／PHI・セキュリティ・データ整合性）を反映した確定版である。
比喩・平易な言い換え・抽象的な言い換えは用いず、関数名・フィールド名・ファイル名・データ構造で記述する。

---

## 実装状況（2026-04 時点）

**Phase 1〜2 を実装済み**。DB→context→看護サマリ生成の一気通貫が動作する（in-memory SQLite で検証）。
実装ファイルと本設計の対応:

| 設計 | 実装ファイル | 状態 |
|---|---|---|
| 中間表現 `ClinicalRecord` 等 | `adapters/models.py` | 実装済み |
| 取得アダプタ抽象 + registry | `adapters/base.py` | 実装済み |
| SQL 取得 | `adapters/sql_source.py`（SQLAlchemy、engine 注入可） | 実装済み |
| 正規化（role別マッピング・コード解決） | `adapters/normalizer.py` | 実装済み |
| サンプリング（extremes） | `adapters/sampler.py` | 実装済み |
| PHI マスク（regex・非可逆） | `adapters/phi_masker.py` | 実装済み（NER は将来オプション） |
| Markdown 化 | `adapters/markdown_renderer.py` | 実装済み |
| 統合 `build_context_from_db` | `adapters/pipeline.py` | 実装済み |
| query_spec ローダー（Pydantic + bind 検証） | `query_specs_loader/{models,loader}.py` | 実装済み |
| query_spec / codesystem YAML | `query_specs/sql_sample.yaml`, `query_specs/codesystems/medis_obs.yaml` | サンプル実装済み |
| `/ingest` エンドポイント | `app.py` | 実装済み（認証は未実装＝ユーザー方針） |
| FHIR / SS-MIX2 取得 | — | 未実装（Phase 3、retrieval 層差し替えで対応） |

未確定事項（§9 の17項目、特に実DBのカラム名・コード体系・看護記録の格納先）はカラム確定時に query_spec / codesystem の更新で対応する。実行時フローは [data-flow.md](data-flow.md) §4 を正とする。

---

## 0. 既存コードの確定事実（接続先）

実装着手前に確認した事実。設計はこれに接続する。

| 確認対象 | ファイル | 確定事実 |
|---|---|---|
| 正規化Markdown形式 | `utils/preprocess.py` | `# 患者ID: <id>` ヘッダ → `- YYYYMMDD`（日付行）→ `  - カルテ#N`（2スペース）→ 本文（4スペースインデント）の階層 |
| 日付チャンク分割 | `graph/nodes/input_adapter.py` | `re.compile(r"^- (\d{8})\s*$", re.MULTILINE)` で分割。`- YYYYMMDD` 単独行が境界。マッチ無しは `_split_by_tokens` にフォールバック |
| チャンクメタデータ | 同 | `chunk_index` は `{"date": "YYYYMMDD", "index": i}` の list |
| 検索の実態 | `graph/nodes/search.py` | `_execute_keyword_search` は `keyword_lower in c.lower()`（**チャンク本文全体の部分文字列マッチ**）。`_execute_date_range_search` は `start_date <= date <= end_date`（**文字列比較**）。結果は **`max_results = 5` チャンクに切られる** |
| 検索の粒度 | 同 | 検索はチャンク（=日付）単位。行単位・項目単位の抽出は行われない。`role`/`label`/`category` は本文に文字列として現れない限り検索に使われない |
| 日付範囲の算出 | 同 | `date_range` は `sorted(set(c["date"] ...))` の最小・最大。`SEARCH_PROMPT` に提示され LLM のツール引数推論に影響 |
| セクション駆動 | `input_adapter.py` / `templates/*.yaml` | テンプレートの `sections[].key/name/description` から `search_plan` を生成。`search_queries` は `[]` 初期化で LLM が `plan` ノードで生成 |
| /ask 入力 | `app.py` | `AskRequest{context: str, patient_id, template_id}`。`context` に正規化済みMarkdownを渡す契約 |
| initial_state | `app.py` | グラフ初期状態は 20 以上のフィールドを持つ（`_search_results`, `_section_sufficient`, `draft_summary`, `reflection_*`, `max_iterations`, `error` 等） |
| 設定機構 | `config/settings.py` | 環境変数ベース。`_resolve()` で `NAME_<ENV>` → `NAME` → default の優先解決。**`config/` に YAML ローダーは存在しない** |
| YAMLローダーの前例 | `templates_loader/loader.py` | YAML 設定は `templates/`（データ）+ `templates_loader/`（ローダー）の系統。`config/` ではない |
| チャンク定数 | `config/settings.py` | `CHUNK_SIZE=130000`、`num_ctx=8192`。`input_adapter._split_by_tokens` は `min(CHUNK_SIZE, 4096)` を使用 |
| 依存 | `pyproject.toml` | SQLAlchemy / FHIR クライアント / PHI ライブラリ（Presidio 等）は未導入 |

**結論**: 既存パイプラインは「正規化済みMarkdown文字列」を `AskRequest.context` で受け取る契約。新規設計の責務は **DB → 正規化済みMarkdown文字列の生成**である。
ただしレビューにより、「`input_adapter` 以降は完全に無改修」という単純な主張は成立しないことが判明した（後述 §5・§6・§9 の小改修が必要）。

---

## 1. 全体方針

```
[取得元: SQL / FHIR / SS-MIX2]
   │  ① RecordSourceAdapter.fetch(patient_id, encounter_id, retrieval_spec)
   ▼
[生レコード: dict[record_category, list[dict]]（取得元固有スキーマ）]
   │  ② RecordNormalizer.normalize(raw, logical_spec)
   ▼
[中間表現: NormalizedRecordSet（list[ClinicalRecord]、取得元非依存）]
   │  ③ Sampler.reduce(record_set, sampling_spec)   ← 時系列間引き（独立ステップ）
   ▼
[間引き済み NormalizedRecordSet]
   │  ④ PhiMasker.mask(record_set)                  ← PHI処理（必須通過点）
   ▼
   │  ⑤ MarkdownRenderer.render(record_set) → (summary_header, dated_markdown)
   ▼
[サマリヘッダ + 日付チャンクMarkdown]
   │  ⑥ build_context() が両者を結合し context 文字列を構築
   ▼
[ _run_graph(context, patient_id, template_id) ]   ← /ask と /ingest 共通
   ▼
[既存パイプライン: input_adapter → Agentic Search → synthesize → reflect → 出力]
```

責務分離（各ステップは単一責任）:

| ステップ | コンポーネント | 責務 |
|---|---|---|
| ① 取得 | `RecordSourceAdapter`（抽象） | 取得元プロトコルの差異を吸収。出力は `{record_category: [row, ...]}` に統一 |
| ② 正規化 | `RecordNormalizer` | 列→意味役割（role）変換、コード→名称解決、欠損処理。`ClinicalRecord` 生成（1行=1レコード） |
| ③ サンプリング | `Sampler` | 同一項目の時系列を間引く（トークン量制御）。Normalizer から分離（M-3 反映） |
| ④ PHI | `PhiMasker` | 構造化列除外（phiフラグ）＋自由記述のマスク。**任意ではなく固定通過点**（A-1 反映） |
| ⑤ Markdown化 | `MarkdownRenderer` | `- YYYYMMDD` 形式への直列化＋サマリヘッダ生成 |
| ⑥ 結合・実行 | `build_context` / `_run_graph` | context 構築とグラフ実行。`/ask` と共通化 |

シリアライズ形式は **Markdown**（日付セクションヘッダ + `項目名: 値` 行 + 自由記述本文）を採用する。
これは `input_adapter` の `- YYYYMMDD` 契約に整合させるため、および調査結果「section headers で区切る Markdown 構造」「`項目名: 値` テンプレート」に整合するため。
（Markdown と raw JSON の優劣は日本語データでの A/B 検証で最終決定する。§10 #14）

---

## 2. クエリ仕様の抽象化（カラム未確定でも動く仕組み）

カラム未確定でも動かすため、取得する列・列の意味役割・コード解決・取得方法を設定ファイル（YAML）で外出しする。
列追加は YAML 更新のみで対応する。

### 2.1 query_spec を「論理層」と「retrieval層」に二分する（指摘3 反映）

`query_spec` を、取得元非依存の **論理層**と、取得元固有の **retrieval層**に明確に分離する。
Normalizer / Sampler / Renderer は論理層のみに依存し、SQL/FHIR/SS-MIX2 の差異は retrieval層に閉じる。

```yaml
# query_specs/sql_default.yaml
spec_id: sql_default
source_type: sql                       # sql / fhir / ssmix2（retrieval層の実装選択）
description: "汎用RDB電子カルテからの看護サマリー入力取得定義"

keys:                                  # 患者・入院の絞り込みキー（共通）
  patient_id:   { required: true }
  encounter_id: { required: true }

records:
  - record_category: vital_sign        # === 論理層（取得元非依存）===
    contributes_to: [nursing_process, medical_equipment]
    columns:
      datetime:  { role: datetime, datetime_kind: measured }  # 測定日時を日付軸に
      item:      { role: item, codesystem: medis_obs@2024 }   # コード→名称（版指定）
      value:     { role: value }
      unit:      { role: unit }
      recorder:  { role: recorder }
    sampling: { strategy: extremes, points: [first, last, min, max] }  # §3 / 指摘5

    retrieval:                          # === retrieval層（source_type 固有）===
      sql: |
        SELECT measured_at AS datetime, item_code AS item, value, unit,
               recorder_name AS recorder
        FROM vital_signs
        WHERE patient_id = :patient_id AND encounter_id = :encounter_id
          AND latest_flag = 1 AND status <> 9      -- 履歴・削除除外
        ORDER BY measured_at ASC

  - record_category: nursing_problem
    contributes_to: [risks, nursing_process]
    columns:
      item:    { role: item, label: 看護問題 }
      goal:    { role: text, label: 目標 }
      tp_plan: { role: text, label: "T-P" }
      ep_plan: { role: text, label: "E-P" }
      status:  { role: subtype, label: 転帰 }   # active/resolved を明示（指摘4）
    retrieval:
      sql: |
        SELECT problem_name AS item, goal, tp_plan, ep_plan, status
        FROM nursing_care_plans
        WHERE patient_id = :patient_id AND encounter_id = :encounter_id
          AND status IN ('active', 'resolved')    -- 解決済みも取得（指摘4）
```

FHIR の場合、同じ論理層（`record_category` / `columns(role)` / `contributes_to` / `sampling`）を保ち、
`retrieval` ブロックのみ差し替える:

```yaml
    retrieval:
      fhir_search:
        resource: Observation
        params: { category: vital-signs, patient: "{patient_id}", encounter: "{encounter_id}" }
        field_map: { datetime: effectiveDateTime, item: code, value: valueQuantity.value, unit: valueQuantity.unit }
```

### 2.2 列→意味役割（role）の語彙

| role | 意味 | Renderer での扱い |
|---|---|---|
| `datetime` | 日付/日時軸（`datetime_kind`: recorded/measured/performed） | チャンクの `- YYYYMMDD` 境界決定に使用 |
| `item` | 観察項目・行為名（コード化されうる） | `codesystem` 指定時はマスタ名称に解決 |
| `value` | 測定値・結果値 | `unit` と連結 |
| `unit` | 単位 | `value` の後ろに付与 |
| `text` | 自由記述（SOAP本文等） | 本文ブロックとして展開。`label` で見出し |
| `subtype` | 記録様式区分・転帰等 | 行ヘッダ・注記に表示 |
| `recorder` | 記録者 | 行末 `（記録者: ...）` |
| `link` / `id` | 結合キー | Markdown非出力。Normalizer/Sampler の結合に使用 |
| `phi` | 個人識別情報（氏名・住所等） | §7 により既定で Markdown 出力から除外 |

### 2.3 コード解決マッピング（版・施設スコープ対応／指摘7 反映）

```yaml
# query_specs/codesystems/medis_obs@2024.yaml
codesystem_id: medis_obs
version: "2024"                 # 版差を吸収
oid: urn:oid:1.2.392.200119.4.804
scope: common                   # common / hanwa / shinkinen（施設ローカルコードは施設スコープ）
codes:
  "31001368": 体温
  "31001848": 収縮期血圧
  "31001849": 拡張期血圧
  "31000001": SpO2
compose_rules:                  # 階層合成（第3階層行為+第4階層修飾語を1項目に）
  - join: ["31006470", "31006471"]   # 例: 清拭 + 部分介助 → "清拭（部分介助）"
    as: "{0}（{1}）"
```

解決順は `HOSPITAL` 環境変数（`config/settings.py`）に従い `scope: <hospital>` → `scope: common` の順。
未知コードは `'unknown'` ではなく `[未解決コード:<code>]` として残し、欠損と区別する。

### 2.4 query_spec のスキーマ検証（M-1 反映）

`config/` に YAML ローダーが無いため、既存の `templates_loader` パターンを踏襲した新規ローダーを設ける。
配置は `query_specs/`（データ）+ `query_specs_loader/`（ローダー）とし、起動時に Pydantic で検証する。

```python
# query_specs_loader/models.py
from pydantic import BaseModel

class ColumnSpec(BaseModel):
    role: str
    label: str | None = None
    codesystem: str | None = None
    datetime_kind: str | None = None  # recorded / measured / performed

class RecordSpec(BaseModel):
    record_category: str
    contributes_to: list[str] = []
    columns: dict[str, ColumnSpec]
    sampling: dict | None = None
    retrieval: dict                    # source_type 固有（sql/fhir_search/ssmix2_path）

class QuerySpec(BaseModel):
    spec_id: str
    source_type: str
    keys: dict[str, dict]
    records: list[RecordSpec]
```

ローダーは `retrieval` 内のバインド変数を `:patient_id` / `:encounter_id` のみに制限し、それ以外を検出したら起動時に例外とする（A-3 反映、SQLインジェクション防止）。

---

## 3. 中間表現（正規化スキーマ）

取得元非依存の Pydantic モデル。`adapters/models.py` に配置。

```python
"""取得元非依存の中間表現（正規化スキーマ）。

SQL / FHIR / SS-MIX2 いずれの取得元から得た行も ClinicalRecord へ正規化する。
Sampler / MarkdownRenderer はこのモデルのみに依存し取得元スキーマを知らない。
"""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class RecordCategory(str, Enum):
    """記録区分。query_spec の record_category と一致させる。"""
    PATIENT_PROFILE = "patient_profile"
    ENCOUNTER = "encounter"
    VITAL_SIGN = "vital_sign"
    NURSING_NOTE = "nursing_note"
    NURSING_PROBLEM = "nursing_problem"
    MEDICATION = "medication"
    LAB_RESULT = "lab_result"
    PROCEDURE = "procedure"
    ADL = "adl"
    RISK_ASSESSMENT = "risk_assessment"
    NURSING_ACUITY = "nursing_acuity"      # 看護必要度A/B/C（指摘4）
    ALLERGY = "allergy"
    INFECTION = "infection"
    INCIDENT = "incident"                  # 転倒転落・急変（指摘4）
    DISCHARGE_SUPPORT = "discharge_support"


class RecordField(BaseModel):
    """正規化レコード内の1項目（列1つに対応）。

    Attributes:
        label: 表示名（コード解決後の項目名 or query_spec の label）。
        value: 値。欠損時は None。Renderer で "記録なし" に変換。
        unit: 単位（数値項目のみ）。
        is_text: 自由記述か（True なら Markdown 本文ブロックへ）。
    """
    label: str
    value: str | None = None
    unit: str | None = None
    is_text: bool = False


class ClinicalRecord(BaseModel):
    """正規化された1記録。

    Attributes:
        event_date: 日付軸（YYYYMMDD への変換元）。None は日付不明。
        date_kind: event_date の意味（recorded/measured/performed）。
        category: 記録区分。
        subtype: 記録様式区分・転帰等（任意）。
        fields: 項目群（順序保持）。
        recorder: 記録者名。
        problem_id: 看護問題ID（看護記録と看護計画の結合キー）。
        cross_cutting: 日付に紐づかない患者横断情報か（§4 サマリヘッダ行き）。
    """
    event_date: datetime | None = None
    date_kind: str | None = None
    category: RecordCategory
    subtype: str | None = None
    fields: list[RecordField] = Field(default_factory=list)
    recorder: str | None = None
    problem_id: str | None = None
    cross_cutting: bool = False


class NormalizedRecordSet(BaseModel):
    """1患者・1入院分の正規化済みレコード集合。

    Attributes:
        patient_id: 患者ID（ローカル）。
        encounter_id: 入院ID。
        records: 全 ClinicalRecord（カテゴリ・日付混在）。
        missing_categories: 取得を試みたが0件だった記録区分（欠損明示用）。
    """
    patient_id: str
    encounter_id: str
    records: list[ClinicalRecord] = Field(default_factory=list)
    missing_categories: list[RecordCategory] = Field(default_factory=list)
```

### 3.1 正規化の規則（Normalizer）

1. `RecordSpec.columns` の `role` に従い各行を `ClinicalRecord` へ変換（1行=1レコード）。
2. `role=item` かつ `codesystem` 指定の列は §2.3 のマップで名称解決。`compose_rules` があれば複数コードを合成。
3. `role=datetime` の値を `event_date` に格納し、`datetime_kind` を `date_kind` に保持。複数 datetime 列がある場合は spec の `datetime_kind` 優先順位に従う（指摘8）。
4. 患者横断情報（`allergy`, `infection`, `nursing_problem` リスト, `patient_profile`, `nursing_acuity` の代表）は `cross_cutting=True` を立てる（§4 でサマリヘッダへ）。
5. 結果値が画像/音声/外部参照型の行は本文から除外し `[添付あり:<種別>]` の注記フィールドのみ残す。
6. 取得0件の `record_category` を `missing_categories` に記録。
7. 文字正規化（Unicode NFKC、改行 `\r\n`→`\n`、機種依存文字置換）を前段で実施。

### 3.2 日付・タイムゾーンの規則（指摘8 反映）

- `event_date`（`datetime`）→ `YYYYMMDD` 変換は JST 固定、日付境界は暦日（00:00–23:59）とする。
- チャンク軸に使う日付は `date_kind` で記録区分ごとに選ぶ（バイタルは measured、処置は performed、記録は recorded）。
- 規則は §4.1 にも明記し、深夜帯記録の前日/翌日振り分けを一意化する。

---

## 4. Markdown化とサマリヘッダ（検索の実態を踏まえた設計）

レビュー C-1/C-2/B-1/B-2/指摘1/指摘2 により、検索が「チャンク本文の部分文字列マッチ・最大5チャンク」である事実を踏まえ、出力を2領域に分ける。

### 4.1 出力2領域

1. **サマリヘッダ領域（検索に依存させず常時供給）**:
   日付に紐づかない、または常に全セクションで必要な患者横断情報（基本属性の非PHI部分、アレルギー、感染症、看護問題リスト、看護必要度、入院日・主病名）を、
   日付チャンクの**前**に置く。`- YYYYMMDD` 行を持たないため `input_adapter` の最初の日付行までは1チャンクに含まれる扱いになり、
   かつ Agentic Search の検索ヒットに依存せず synthesize に渡る経路を §6 の小改修で確保する。
   これにより「アレルギー・感染症が `max_results=5` 圏外に落ちて欠落する」事故（指摘2）を防ぐ。

2. **日付チャンク領域**: `event_date` を持つ経過情報を `- YYYYMMDD` 単位で出力。

`- 00000000` 擬似日付は使用しない（C-1/B-1: `date_range` の下限を汚染し `_execute_date_range_search` の文字列比較を歪めるため）。
日付不明の経過記録は、サマリヘッダ領域の「日付不明の記録」小見出しにまとめる。

### 4.2 変換規則（`MarkdownRenderer.render(record_set) -> str`）

1. ヘッダ行 `# 患者ID: {patient_id}`。
2. `cross_cutting=True` のレコードを「サマリヘッダ領域」として最初の日付行の前に出力。
   見出し語彙は、テンプレートのセクション `description` に現れる語（「医療機器」「処置部位」「指導」「継続」「リスク」等）を意図的に含める（C-2: 検索ヒットは本文語彙が決めるため）。
3. 残りを `event_date`（`date_kind` で選択）昇順でソートし、同一 `YYYYMMDD` を1日付チャンクに集約。チャンク境界は `- YYYYMMDD` 単独行（`input_adapter` 契約）。
4. 各日付チャンク内: `category` ごとに `  - <記録区分日本語名>` 小見出し（区分名はセクション検索語と一致する日本語を選ぶ）。
5. `fields`:
   - 構造化値（`is_text=False`）: `    {label}: {value}{unit}` の1行（4スペースインデント）。
   - 自由記述（`is_text=True`）: `    {label}:` の後に本文を4スペースインデントで展開。
6. `recorder` があれば行末 `（記録者: {recorder}）`。
7. 欠損は `記録なし` と明示（NULL/空文字を渡さない）。`missing_categories` はサマリヘッダ領域に `（記録なし: <区分>）` として出力。

### 4.3 チャンク数と `max_results=5` の整合（B-2 反映）

長期入院では日付チャンク数が `max_results=5` を超え、中間経過が検索で落ちる。対策を `sampling` に持たせる:

- `chunk_granularity`: `day`（既定）/ `week` / `phase`（入院期・経過期・退院期）。長期入院では週・期単位に粗くしてチャンク数を抑える。
- 各日付チャンクの想定トークン量 × チャンク数が `num_ctx`（現 8192）に収まるかを Phase 1 の検証項目とする。
- この整合は §10 #10 の確定対象。

### 4.4 出力例

```markdown
# 患者ID: 0311

## サマリ基本情報（全期間共通）
- 入院日: 20230209 / 主病名: 誤嚥性肺炎
- アレルギー: 記録なし
- 感染症: MRSA 陽性（接触予防策の継続が必要・リスク）
- 看護問題リスト:
    #1 ガス交換障害（転帰: 継続）
    #2 誤嚥リスク状態（転帰: 解決）
- 看護必要度: A項目2点 B項目3点（20230209時点）

- 20230209
  - バイタルサイン
    体温: 37.8℃
    収縮期血圧: 138mmHg
    SpO2: 94%（記録者: 看護師A）
  - 看護記録
    S(主観): 息苦しいと訴えあり
    O(客観): 右下肺野で湿性ラ音聴取
    A(評価): 喀痰排出不十分、呼吸状態悪化のリスク
    P(計画): 体位ドレナージ継続、SpO2モニタ（記録者: 看護師A）

- 20230215
  - バイタルサイン
    体温: 36.6℃
    SpO2: 98%（記録者: 看護師B）
  - 看護記録
    記録: 自力歩行訓練開始。ふらつきなし（記録者: 看護師B）
```

---

## 5. テンプレートのセクションとの対応（疎結合）

セクションとの対応は query_spec の `contributes_to`（記録区分→セクションkey）で表現し、コードにハードコードしない。

ただしレビュー C-2/指摘6 により、`contributes_to` は現状のパイプラインに一切流れない（検索は本文文字列マッチで、`search_queries` は `[]` 初期化）。
そこで本設計は `contributes_to` を**実際に効かせる**ため §6 の小改修を組み込む。

| 記録区分 | hanwa セクション (key) への寄与 |
|---|---|
| `nursing_note`(E-P実施) | `instruction` |
| `medication`, `procedure`, `vital_sign`(機器系) | `medical_equipment` |
| `vital_sign`, `nursing_note`, `adl`, `nursing_problem`, `nursing_acuity` | `nursing_process` |
| `nursing_note`(S/A本文), `discharge_support` | `patient_condition` |
| `nursing_problem`, `risk_assessment`, `allergy`, `infection`, `incident` | `risks` |
| 上記未分類 | `others` |

---

## 6. 既存コードへの追加/変更（最小・レビュー反映）

### 6.1 新規ファイル

| ファイル | 役割 |
|---|---|
| `adapters/models.py` | §3 中間表現 Pydantic |
| `adapters/base.py` | §7 抽象基底 `RecordSourceAdapter` + ファクトリ |
| `adapters/sql_source.py` | SQL 実装（SQLAlchemy） |
| `adapters/fhir_source.py` | FHIR 実装（後続フェーズ） |
| `adapters/ssmix2_source.py` | SS-MIX2 実装（後続フェーズ） |
| `adapters/normalizer.py` | §3.1 生レコード→ClinicalRecord |
| `adapters/sampler.py` | §3 時系列間引き（独立ステップ、M-3 反映） |
| `adapters/phi_masker.py` | §7 PHI処理（固定通過点、A-1 反映） |
| `adapters/markdown_renderer.py` | §4 ClinicalRecord→Markdown（サマリヘッダ含む） |
| `adapters/pipeline.py` | `build_context(patient_id, encounter_id, spec_id) -> str` |
| `query_specs/*.yaml` | §2 クエリ定義 |
| `query_specs/codesystems/*.yaml` | §2.3 コード解決 |
| `query_specs_loader/loader.py`, `models.py` | §2.4 ローダー＋Pydantic検証 |

### 6.2 既存ファイルへの変更

- `config/settings.py`: DSN 等の環境変数（例 `EHR_DB_DSN`）と `QUERY_SPEC_DIR` を追加。既存定数は不変更。
- `app.py`: `_run_graph()` を抽出し `/ask` と新規 `/ingest` で共有（C-3 反映）。

```python
def _run_graph(context: str, patient_id: str, template_id: str | None) -> AskResponse:
    """context を受けてグラフを実行し AskResponse を返す共通処理。

    initial_state（20以上のフィールド）の構築をここに一元化し、
    /ask と /ingest の重複と初期化漏れを防ぐ。
    """
    ...

class IngestRequest(BaseModel):
    patient_id: str
    encounter_id: str
    query_spec_id: str = "sql_default"
    template_id: str | None = None

@app.post("/ingest", response_model=AskResponse)
async def ingest(req: IngestRequest) -> AskResponse:
    """DBから取得・正規化・Markdown化して看護サマリーを生成する。"""
    # 認証・patient_id/encounter_id の入力検証（§7）
    context = build_context(req.patient_id, req.encounter_id, req.query_spec_id)
    return _run_graph(context, req.patient_id, req.template_id)
```

- `graph/nodes/input_adapter.py`: **小改修**（C-2/指摘6 反映）。
  テンプレートの `search_plan` 各セクションの `search_queries` を `[]` のままにせず、
  query_spec の `contributes_to` を逆引きして「該当セクションに寄与する記録区分の日本語名・項目ラベル」を初期投入する。
  これにより `plan` ノードの JSON 生成が失敗しても（既知の不具合）、`search.py` のフォールバック検索（`search_queries` 利用）が空振りしない。

### 6.3 段階的移行

1. **Phase 1（接続検証）**: `models.py` + `markdown_renderer.py` + `sql_source.py`（最小カラム）+ `pipeline.build_context`。
   Renderer 出力を `/ask` に手動投入し、`input_adapter` が日付分割すること、サマリヘッダが synthesize に渡ることをテストで確認（§4.3 のトークン量検証含む）。
2. **Phase 2（自動化）**: `/ingest` + `_run_graph` 抽出。query_spec/codesystem ローダーと Pydantic 検証。Normalizer・Sampler・`contributes_to` の search_queries 注入。
3. **Phase 3（取得元拡張）**: `fhir_source.py` / `ssmix2_source.py`。retrieval層差し替えのみで論理層不変を確認。
4. **Phase 4（PHI・本番）**: `phi_masker.py` を固定通過点として組込み、認証・監査ログを実装、閉域で実データ検証・看護師レビュー。

各フェーズで Definition of Done（既存動作非破壊・Critical ゼロ・docstring/型）を満たす。

---

## 7. 接続方式・PHI・セキュリティ・データ整合性

### 7.1 取得アダプタ（DB非依存）

```python
# adapters/base.py
from abc import ABC, abstractmethod

class RecordSourceAdapter(ABC):
    """記録取得アダプタの抽象基底。取得とプロトコル解釈のみを担う。"""

    @abstractmethod
    def fetch(self, patient_id: str, encounter_id: str,
              retrieval_specs: dict) -> dict[str, list[dict]]:
        """記録区分ごとの生レコードを取得する。

        Args:
            patient_id: 患者ID（ローカル）。
            encounter_id: 入院ID。
            retrieval_specs: record_category -> retrieval ブロック（source_type固有）。

        Returns:
            {record_category: [row_dict, ...]}。取得0件はキーごと空リスト。

        Raises:
            ConnectionError: 取得元への接続失敗。
        """
        raise NotImplementedError

_REGISTRY: dict[str, type[RecordSourceAdapter]] = {}

def register_adapter(source_type: str):
    def _wrap(cls):
        _REGISTRY[source_type] = cls
        return cls
    return _wrap

def get_adapter(source_type: str) -> RecordSourceAdapter:
    if source_type not in _REGISTRY:
        raise KeyError(f"未登録の取得元: {source_type}")
    return _REGISTRY[source_type]()
```

取得元の切替は `source_type` のみ。Normalizer/Sampler/Renderer/既存パイプラインは無変更。
（論理層と retrieval層の分離により、この主張が retrieval層を除いて成立する。指摘3 反映）

### 7.2 PHI（A-1/A-2 反映: 2レイヤー必須）

PHI 処理は**任意フックではなく `pipeline` の固定通過点**とし、2レイヤーで防御する:

1. **構造化列の除外**: `role: phi` の列（氏名・住所・電話・KP・医師名・施設名）は Renderer 出力対象から除外（既定）。
2. **自由記述のマスク**: SOAP/DAR 等 `role=text` 本文に埋め込まれた氏名等を regex（電話・郵便番号・日付以外の数字列）+ 任意で NER（将来オプション、依存追加が必要なため Phase 4 で評価）でマスク。

マスクは非可逆の固定マスク（`[患者名]` 等）を第一候補とし、対応表を作らない（A-2: 対応表のログ漏洩を回避）。
マスク前 Markdown は変数に保持し続けず、例外ハンドラで request body・context 文字列をログ出力しない（設計制約）。

### 7.3 セキュリティ（A-3/A-4 反映）

- DB接続情報は YAML に書かず `connection_ref` で識別、実値は環境変数（`_resolve`）。`.env` は `.gitignore` 対象。
- SQL は bindparams（`:patient_id`/`:encounter_id`）強制。`:` バインド変数は2種のみ許可をローダーで保証（§2.4）。query_spec は読み取り専用の配布物として扱う。
- `/ingest` は **認証必須**。`patient_id`/`encounter_id` は正規表現（英数字等）で入力検証。
- **監査ログ**（3省2ガイドライン相当）: who / when / patient_id を記録（PHI本体は記録しない）。§10 に確定項目として追加。

### 7.4 データ整合性（B-1/B-3 反映）

- 不明日付: `- 00000000` 擬似日付を使わず、サマリヘッダ領域の「日付不明の記録」へ（§4.1）。`date_range` 汚染を回避。
- 履歴・改ざん不可: retrieval層の WHERE で `latest_flag=1 AND status<>削除` を強制（JAHIS 3層モデル）。
- 複数取得元併用時の重複排除・突合は、`source_ref`（取得元参照）と `record_category`+`event_date`+`item` のキーで Normalizer 後に dedupe する（Phase 3 で詳細化）。

---

## 8. 取得すべき入力データ項目の候補一覧（確定ではなく候補）

調査結果＋レビュー指摘4で補強。各項目に `record_category` と寄与セクションを対応づける。確定は §9 のカラム確定フェーズで実施。

| 候補項目 | record_category | 寄与セクション(hanwa) | 取得元の典型 |
|---|---|---|---|
| 患者基本（氏名・生年月日・性別・住所・KP） | patient_profile | （PHI、§7で原則マスク/除外） | 患者マスタ / FHIR Patient |
| 入院日・退院日・病棟・主治医・退院先区分 | encounter | nursing_process | 入院履歴 / FHIR Encounter |
| 主病名・現病歴・既往歴 | encounter | nursing_process, patient_condition | FHIR Condition(ICD10) |
| アレルギー（薬物・食物・反応） | allergy | risks, medical_equipment | FHIR AllergyIntolerance |
| 感染症（HBs/HCV/MRSA/結核） | infection | risks | FHIR Condition |
| バイタル（体温/脈/血圧/SpO2/呼吸/意識JCS・GCS） | vital_sign | nursing_process, medical_equipment | フローシート / FHIR Observation(MEDIS) |
| 内服薬（薬剤名・用法用量・管理方法・副作用） | medication | medical_equipment | 処方 / FHIR MedicationRequest(HOT) |
| 検査値（血液/尿・JLAC10） | lab_result | nursing_process | FHIR Observation(JLAC10) |
| ADL各項目自立度・食事形態・嚥下・排泄・移動補助具 | adl | nursing_process | 施設固有スコアテーブル |
| **看護必要度（A/B/C項目）** | nursing_acuity | nursing_process, risks | 看護必要度テーブル（指摘4で追加） |
| 看護問題リスト（NANDA-I等）・優先順位・**転帰(解決/未解決/継続)** | nursing_problem | risks, nursing_process | 看護計画 / FHIR CarePlan（active+resolved、指摘4） |
| 看護計画 O-P/T-P/E-P | nursing_problem | instruction, nursing_process | 看護計画テーブル |
| 経過記録 SOAP/DAR本文 | nursing_note | nursing_process, patient_condition | 看護記録テーブル(LOB) / 拡張ストレージ |
| 指示（種別・頻度・指示者）・実施（処置・吸引・経管栄養・ストーマ・点滴・酸素） | procedure | medical_equipment | JAHIS 指示・実施情報 / FHIR Procedure |
| リスク評価（転倒転落・褥瘡DESIGN-R/ブレーデン） | risk_assessment | risks | 施設固有アセスメント |
| 認知(HDS-R/MMSE)・鎮静(RASS)・疼痛(NRS/VAS) | risk_assessment / vital_sign | risks, patient_condition | FHIR Observation / 施設固有 |
| **インシデント/イベント（転倒・急変）** | incident | risks | インシデント記録（指摘4で追加） |
| **退院時バイタル・退院時ADL・退院時状態像** | vital_sign / adl | nursing_process | サンプリング last + 退院時アセス（指摘4） |
| 退院指導内容・患者家族の理解度 | nursing_note(E-P) | instruction, patient_condition | 看護記録(E-P実施) |
| 社会資源（介護保険・要介護度・ケアマネ・サービス）・退院後継続課題 | discharge_support | risks, others | 退院支援記録 / 地域連携API / 入力フォーム |
| 作成者情報 | patient_profile | others | システム自動付与 |
| **前回入院サマリ・外来経過**（再入院時） | （スコープ要確認） | — | 過去サマリ（§10 #15、現状スコープ外候補） |

---

## 9. 未確定事項・要確認事項（カラム確定時に決める）

| # | 要確認事項 | 確定の影響先 |
|---|---|---|
| 1 | 取得元の実体（RDB直接 / FHIR / SS-MIX2、単一/複数） | `source_type`・最初に実装するアダプタ |
| 2 | patient_id・encounter_id の実カラム名・型・入院IDの有無 | `keys`・JOIN・入力検証 |
| 3 | バイタル等のコードがローカルか MEDIS標準か（マスタ種別） | `codesystem` の要否・粒度・版 |
| 4 | 看護記録(SOAP/自由記述)の格納先(LOB/拡張ストレージ/API) | `nursing_note` の取得経路 |
| 5 | 看護記録↔看護計画のリンクキー(problem_id相当)の有無 | 記録と計画の結合可否 |
| 6 | 履歴・削除フラグの列名(latest_flag/status相当) | WHERE 条件・改ざん履歴除外 |
| 7 | ADL/褥瘡/疼痛/看護必要度の保持形式(コード/フリーテキスト/スコア) | 正規化規則 |
| 8 | 退院先区分のコード体系 | 将来のテンプレート動的切替(急性期/回復期/在宅) |
| 9 | 社会資源情報の所在(院内DB/地域連携API/手入力) | `discharge_support` の取得手段 |
| 10 | 代表値サンプリング戦略・チャンク粒度・トークン上限と num_ctx(現8192)/max_results(現5) の整合 | `sampling`・`chunk_granularity`（§4.3） |
| 11 | PHIを本文に含める運用要否・マスク方式(非可逆/可逆) | `phi`フラグ既定・Masker 構成 |
| 12 | 結果値に画像/音声/外部参照が混在するか | バイナリ除外ロジックの要否 |
| 13 | 文字コード(cp932/UTF-8)・機種依存文字の実態 | Normalizer 前段の正規化 |
| 14 | シリアライズ形式(Markdown vs raw JSON)の日本語A/B検証 | Renderer 出力形式 |
| 15 | 前回入院サマリ・外来経過をスコープに含めるか | 取得対象・再入院時の扱い |
| 16 | `/ingest` の認証方式・監査ログ要件(3省2ガイドライン) | 認証層・監査ログ実装（§7.3） |
| 17 | datetime 列の意味(記録日時/測定日時/実施日時)とTZ | チャンク軸日付の選択（§3.2） |

---

## 10. レビューで確認された既存パイプライン側の制約（本設計の前提）

本設計の有効性は、既存 `search.py` の以下の制約に依存する。これらは入力設計だけでは解消できず、必要なら別途パイプライン改修を要する。

1. **検索はチャンク本文の部分文字列マッチ**（`keyword_lower in c.lower()`）。`role`/`label` 等のメタデータは本文に文字列で現れない限り検索に使われない → Renderer の本文語彙設計が検索ヒットを決める（§4.2-2,4）。
2. **検索結果は `max_results=5` チャンクに制限**。長期入院で日付チャンクが多いと中間経過が落ちる → サマリヘッダ常時供給（§4.1）＋チャンク粒度オプション（§4.3）で緩和。
3. **`_execute_date_range_search` は文字列比較**。`- 00000000` 擬似日付は `date_range` を汚染するため不使用（§4.1）。
4. **`search_queries` は `[]` 初期化で `plan` ノード依存**。`plan` の JSON 生成不具合時にフォールバックが空振りするため、`contributes_to` からの初期投入を `input_adapter` に追加（§6.2）。

これらは「DB入力設計を完璧にしても、パイプライン側の検索上限で完全性が頭打ちになりうる」ことを意味する。
本設計はサマリヘッダ常時供給と検索語彙設計で実務上の完全性を確保するが、根本的にはパイプライン側で
「必須項目を検索に乗せず全セクションへ常時供給する経路」を持つことが望ましい（将来の改修候補）。
