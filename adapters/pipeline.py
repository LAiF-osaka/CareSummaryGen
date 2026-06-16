"""DB入力アダプタの統合パイプライン。

DB取得 → 正規化 → サンプリング → PHI処理 → Markdown化 の順で、
看護サマリ生成パイプラインへ渡す context 文字列（日付チャンク Markdown）を
構築する。

詳細設計: docs/db-input-design.md §1, §6 を参照。
"""

from adapters.base import get_adapter
from adapters.markdown_renderer import render
from adapters.models import NormalizedRecordSet
from adapters.normalizer import normalize
from adapters.phi_masker import mask
from adapters.sampler import sample
from query_specs_loader.loader import load_query_spec


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


def build_context_from_db(
    patient_id: str,
    encounter_id: str,
    spec_id: str,
    *,
    source_adapter=None,
    apply_phi_mask: bool = True,
) -> str:
    """DBから取得・正規化・整形して context 文字列を構築する。

    取得元 → 正規化 → サンプリング → PHIマスク → Markdown化 を順に実行する。

    Args:
        patient_id: 患者ID。
        encounter_id: 入院ID。
        spec_id: query_spec のID。
        source_adapter: 取得アダプタの注入（テスト用）。未指定なら
            spec の source_type からファクトリで生成する。
        apply_phi_mask: 自由記述の PHI マスクを適用するか。

    Returns:
        context として渡す Markdown 文字列。
    """
    spec = load_query_spec(spec_id)
    adapter = source_adapter or get_adapter(spec.source_type)

    raw = adapter.fetch(patient_id, encounter_id, spec)
    record_set = normalize(raw, spec, patient_id, encounter_id)
    record_set = sample(record_set, spec)
    if apply_phi_mask:
        record_set = mask(record_set)

    return render(record_set)
