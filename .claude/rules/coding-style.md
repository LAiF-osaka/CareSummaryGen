---
description: "コードコメント・Docstring スタイル規約（Python: Google Style）"
---

# コメント・Docstring スタイル規約

コードを変更・新規作成するときは、このルールに従ってコメントと Docstring を記述する。

## 共通原則

- **"Why" not "What"**: コードが表現できない理由・制約・トレードオフのみコメントする
- **DRY**: 型情報はコード側に一元化。コメントでは意味・用途を記述する
- **TODO/FIXME には説明必須**: `TODO: 説明` 形式。説明なしの放置 TODO は禁止
- **コメントアウトコードの禁止**: git 履歴で参照する。旧コードをコメントで残さない
- **腐敗コメントの回避**: コード変更時は対応するコメントも必ず更新する
- **句読点・スペル・文法に注意**: コメントはナラティブテキストとして読めるように書く

## Python — Google Style Docstring

参照: https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings

### モジュール docstring（全ファイル必須）

```python
"""セッション状態の永続化リポジトリ。

SQLite / Azure Blob に対してセッション成果物の CRUD を提供する。
StateStore から分離された読み書き責務を担う。

主要クラス:
    SessionRepository: セッション単位の CRUD
    ArtifactRepository: 成果物（PDF / JSON）の保存・取得
"""
```

### クラス docstring（責務 + Attributes）

```python
class SampleClass:
    """Summary of class here.

    Longer class information...

    Attributes:
        likes_spam: A boolean indicating if we like SPAM or not.
        eggs: An integer count of the eggs we have laid.
    """
```

- 1行目は「このインスタンスが何を表すか」を記述する
- `Exception` サブクラスは「何のエラーか」を記述する（発生コンテキストではない）
- `class` であることを繰り返さない（❌ `"""Class that describes..."""`）

### 関数・メソッド docstring（Args / Returns / Raises）

以下のいずれかに該当する関数は docstring 必須:
- 公開 API
- 非自明なロジック
- 小さくないサイズ

```python
def fetch_rows(
    table_handle: smalltable.Table,
    keys: Sequence[bytes | str],
    require_all_keys: bool = False,
) -> Mapping[bytes, tuple[str, ...]]:
    """Fetches rows from a Smalltable.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by table_handle.  String keys will be UTF-8 encoded.

    Args:
        table_handle: An open smalltable.Table instance.
        keys: A sequence of strings representing the key of each table
          row to fetch.  String keys will be UTF-8 encoded.
        require_all_keys: If True only rows with values set for all keys
          will be returned.

    Returns:
        A dict mapping keys to the corresponding table row data fetched.

    Raises:
        IOError: An error occurred accessing the smalltable.
    """
```

- 型ヒントがある場合、Args/Returns から型を省略する（DRY）
- `@override` メソッドは docstring 不要（差分がある場合のみ記述）
- `@property` は属性スタイルで記述する（❌ `"""Returns the path."""`）
- 関数名とシグネチャで自明なら Args/Returns セクションを省略可

### ブロック・インラインコメント

```python
# 加重辞書検索で配列中の i の位置を特定する。
# 配列サイズと最大値から位置を推定し、二分探索で確定する。

if i & (i - 1) == 0:  # True if i is 0 or a power of 2.
```

- **コードの前に**: 複雑な操作の意図を数行で説明
- **行末に**: 非自明な式の意味を補足（コードから2スペース以上空ける）
- **コードを説明しない**: 読者は Python を知っている前提。あなたの「意図」を書く

### コメント記載量の目安

| 対象 | 必須 | 推奨 | 不要 |
|---|---|---|---|
| モジュール | docstring（責務 + 主要クラス/関数） | 設計上の注意・関連ドキュメントリンク | 著者名・日付・変更履歴 |
| クラス | docstring（責務 + Attributes） | Example | 自明なメソッドの docstring |
| 公開関数 | docstring（Args / Returns / Raises） | Example | private 関数の網羅的 docstring |
| Flask エンドポイント | docstring（エンドポイントの概要） | リクエスト/レスポンス形式の記載 | decorator と docstring の重複 |
| テスト関数 | 関数名で意図を表現 | 非自明なテストのみ docstring | 全テストへの網羅的 docstring |

## AI-first 補足ルール

従来の「自明なら省略」に加え、AI エージェントが読者であることを意識する:

- **公開 API は省略しない**: 人間に自明でも AI の盲点になる。最低 1行サマリーを書く
- **モジュール先頭に依存関係・アーキテクチャ上の位置を記述**: AI ナビゲーションヒントとして機能する
- **制約・不変条件を明示**: スレッド安全性、リトライ回数、排他制御等の暗黙知を docstring に書く
- **ただし冗長にしない**: 情報密度を最適化し、人間にも読みやすい簡潔さを維持する
