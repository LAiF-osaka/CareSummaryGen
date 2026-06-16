"""記録取得アダプタの抽象基底とファクトリ。

取得元（SQL / FHIR / SS-MIX2）のプロトコル差異を吸収し、共通の
「record_category -> 行のリスト」インタフェースを提供する。
正規化（コード解決・列マッピング）は Normalizer の責務であり、
本アダプタは取得とプロトコル解釈のみを担う（単一責任）。

詳細設計: docs/db-input-design.md §7 を参照。
"""

from abc import ABC, abstractmethod

from query_specs_loader.models import QuerySpec


class RecordSourceAdapter(ABC):
    """記録取得アダプタの抽象基底。"""

    @abstractmethod
    def fetch(
        self,
        patient_id: str,
        encounter_id: str,
        spec: QuerySpec,
    ) -> dict[str, list[dict]]:
        """記録区分ごとの生レコードを取得する。

        Args:
            patient_id: 患者ID（ローカル）。
            encounter_id: 入院ID。
            spec: 検証済み query_spec。

        Returns:
            {record_category: [row_dict, ...]}。取得0件の区分はキーごと
            空リストで返す（欠損明示のため）。

        Raises:
            ConnectionError: 取得元への接続失敗。
        """
        raise NotImplementedError


_REGISTRY: dict[str, type[RecordSourceAdapter]] = {}


def register_adapter(source_type: str):
    """source_type 名でアダプタ実装を登録するデコレータ。"""

    def _wrap(cls: type[RecordSourceAdapter]) -> type[RecordSourceAdapter]:
        _REGISTRY[source_type] = cls
        return cls

    return _wrap


def get_adapter(source_type: str) -> RecordSourceAdapter:
    """source_type に対応するアダプタ実装を生成する。

    Raises:
        KeyError: 未登録の source_type を指定した場合。
    """
    if source_type not in _REGISTRY:
        raise KeyError(f"未登録の取得元: {source_type}")
    return _REGISTRY[source_type]()
