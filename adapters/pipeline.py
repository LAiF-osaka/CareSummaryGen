"""DB入力アダプタの統合パイプライン。

正規化レコード集合から、看護サマリ生成パイプラインへ渡す
context 文字列（日付チャンク Markdown）を構築する。

詳細設計: docs/db-input-design.md §1, §6 を参照。

Phase 1 では NormalizedRecordSet を直接受け取り Markdown 化する
（接続契約の検証用）。Phase 2 以降で DB取得（RecordSourceAdapter）→
正規化（Normalizer）→ サンプリング（Sampler）→ PHI処理（PhiMasker）→
本関数、の順で前段を追加する。
"""

from adapters.markdown_renderer import render
from adapters.models import NormalizedRecordSet


def build_context(record_set: NormalizedRecordSet) -> str:
    """正規化レコード集合から context 文字列を構築する。

    出力は既存 input_adapter が解釈できる `- YYYYMMDD` 日付チャンク形式の
    Markdown であり、`AskRequest.context` にそのまま渡せる。

    Args:
        record_set: 正規化済みレコード集合。

    Returns:
        context として渡す Markdown 文字列。
    """
    return render(record_set)
