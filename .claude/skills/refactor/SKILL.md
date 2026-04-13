---
name: refactor
description: リファクタリングワークフロー。分割・移動・テストを実施する
argument-hint: "リファクタ対象の説明"
disable-model-invocation: true
---

# Refactoring Workflow

`$ARGUMENTS` に記述されたリファクタリングを完遂してください。

## Step 1: 現状分析

以下を実施:
- `$ARGUMENTS` の対象コードを全文読み込み
- 現在の責務・行数・依存関係を分析
- 分割後のファイル構成案を提示

## Step 2: タスク分解

以下を実施:
- 具体的な移動対象（関数・クラス・定数）のリスト作成
- リファクタリングの段階的計画
- テスト影響範囲の特定

## Step 3: 実装

**実装時の必須ルール**:
1. まず新規ファイルにロジックをコピー → テスト pass 確認
2. 元ファイルを re-export ファサードに変更 → テスト pass 確認
3. import パスを新パスに書き換え → テスト pass 確認
4. ファサード削除 → テスト pass 確認

各ステップで `uv run pytest` を実行。

### デバッグログの埋め込み（実装時に必ず遵守）

リファクタリングで新規作成・移動するコードには、**本番では表示されないデバッグレベルのログ**を積極的に埋め込むこと。
**ロジックだけ移動してデバッグログを付けないのは禁止。**

#### Python — `logger.debug()`

```python
import logging
logger = logging.getLogger(__name__)

# 関数の入口・出口
async def search(self, query: str, top_k: int = 8) -> RAGSearchResult:
    logger.debug("search called: query=%r, top_k=%d", query, top_k)
    ...
    logger.debug("search completed: items=%d", len(result.items))
    return result

# 分岐・フォールバック
if self._retriever is None:
    logger.debug("No primary retriever available, falling back to default")

# 外部呼び出し前後
logger.debug("Calling LLM: model=%s, prompt_len=%d", model, len(prompt))
response = await chain.invoke(...)
logger.debug("LLM response: tokens=%d", response.usage.total_tokens)
```

#### ログ埋め込みの判断基準

| 場面 | 必須 | 推奨 | 不要 |
|---|---|---|---|
| **関数の入口（引数サマリ）** | 公開関数・エンドポイント | private 関数 | 1行のヘルパー |
| **関数の出口（結果サマリ）** | 検索・API呼び出し | 変換・整形 | getter |
| **分岐・フォールバック** | 条件が非自明 | 全ての else/except | 自明な null チェック |
| **外部サービス呼び出し前後** | 全て（API, DB） | — | — |
| **コレクション操作** | フィルタ・重複排除の前後件数 | map/reduce | 単純な append |

#### 命名規約

- **Python**: `logger.debug("メッセージ", arg1, arg2)` — `%s`/`%d`/`%r` フォーマット（f-string 禁止 = 遅延評価）

### コメント規約（実装時に必ず遵守）

リファクタリングで新規作成・移動するコードには、以下のコメント規約を適用すること。
**コードだけ移動してコメントを付けないのは禁止。**

#### 共通原則

- **"Why" not "What"**: コードが表現できない理由・制約・トレードオフのみコメントする
- **DRY**: 型情報はコード側に一元化。コメントでは意味・用途を記述する
- **TODO/FIXME にはチケット番号必須**: `TODO(#123): 説明` 形式。番号なしの放置 TODO は禁止
- **コメントアウトコードの禁止**: git 履歴で参照する。旧コードをコメントで残さない
- **腐敗コメントの回避**: コード変更時は対応するコメントも必ず更新する

#### Python — Google Style Docstring

```python
# ── モジュール docstring（全ファイル必須）──────────────────────
"""看護サマリー生成のリポジトリ。

ChromaDB に対してドキュメントの CRUD を提供する。

主要クラス:
    DocumentRepository: ドキュメント単位の CRUD
"""

# ── クラス / 関数 docstring ────────────────────────────────────
class DocumentRepository:
    """ドキュメント単位の CRUD リポジトリ。

    Attributes:
        collection_name: ChromaDB コレクション名。
    """

    async def get_document(self, doc_id: str) -> Document | None:
        """ドキュメント ID でドキュメントを取得する。

        Args:
            doc_id: ドキュメント識別子。

        Returns:
            ドキュメントが存在すれば Document、なければ None。
        """
```

#### コメント記載量の目安

| 対象 | 必須 | 推奨 | 不要 |
|---|---|---|---|
| **Python モジュール** | docstring（責務 + 主要クラス/関数） | 設計上の注意 | 著者名・日付・変更履歴 |
| **Python クラス** | docstring（責務 + Attributes） | Example | 自明なメソッドの docstring |
| **Python 公開関数** | docstring（Args / Returns / Raises） | Example | private 関数の網羅的 docstring |
| **Flask エンドポイント** | docstring | responses パラメータ | decorator と docstring の重複 |
| **テスト関数** | 関数名で意図を表現 | 非自明なテストのみ docstring | 全テストへの網羅的 docstring |

## Step 4: 整合性チェック（必須）

実装完了後、以下を**全て実行**して結果を確認:

```bash
# 1. ファイルサイズチェック（300行超がないこと）
find . -name "*.py" ! -path "*__pycache__*" ! -path "*.venv*" -exec wc -l {} + | sort -rn | head -15

# 2. テスト実行
uv run pytest

# 3. リント実行
uv run flake8
```

## Step 5: コードレビュー

以下を確認:
- DRY 原則違反がないか
- 後方互換が維持されているか
- **コメント品質チェック**:
  - 新規/移動ファイルにモジュール docstring があるか（Python: Google Style）
  - 公開クラス・関数に docstring があるか
  - インラインコメントが「Why」を説明しているか（「What」の繰り返しがないか）
  - TODO/FIXME にチケット番号が付いているか
  - コメントアウトされたコードが残っていないか
  - 型ヒントと docstring で型情報が重複していないか

## Step 6: コミット & プッシュ

以下を実施:
- `refactor(scope): [説明]` でコミット（Conventional Commits）
- develop にプッシュ
- 必要に応じて PR 作成

## 完了報告

1. 変更前後のファイル構成比較
2. 行数削減の実績
3. テスト結果
4. コメント品質: 新規/移動ファイルの docstring カバレッジ（モジュール / クラス / 公開関数）
