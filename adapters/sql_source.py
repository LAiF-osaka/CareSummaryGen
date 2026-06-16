"""SQL 取得アダプタ。

query_spec の retrieval.sql を :patient_id / :encounter_id バインドで実行し、
record_category ごとの生レコードを返す。SQLAlchemy を用いて DB 非依存に保つ。

接続は環境変数 EHR_DB_DSN（settings 経由）から解決する。テスト用に
engine を注入できる（in-memory SQLite 等）。

詳細設計: docs/db-input-design.md §7, §8.2 を参照。
"""

import os

from sqlalchemy import Engine, create_engine, text

from adapters.base import RecordSourceAdapter, register_adapter
from query_specs_loader.models import QuerySpec


@register_adapter("sql")
class SqlRecordSource(RecordSourceAdapter):
    """汎用RDBから SQLAlchemy で取得する実装。

    Attributes:
        engine: SQLAlchemy Engine。未指定時は EHR_DB_DSN から生成する。
    """

    def __init__(self, engine: Engine | None = None) -> None:
        self._engine = engine

    def _get_engine(self) -> Engine:
        """Engine を取得する（未注入なら DSN から生成）。"""
        if self._engine is not None:
            return self._engine
        dsn = os.environ.get("EHR_DB_DSN")
        if not dsn:
            raise ConnectionError(
                "EHR_DB_DSN が未設定です（SQL 取得元の接続文字列）"
            )
        self._engine = create_engine(dsn)
        return self._engine

    def fetch(
        self,
        patient_id: str,
        encounter_id: str,
        spec: QuerySpec,
    ) -> dict[str, list[dict]]:
        """query_spec の各 SQL を実行して生レコードを返す。"""
        engine = self._get_engine()
        params = {
            "patient_id": patient_id,
            "encounter_id": encounter_id,
        }
        result: dict[str, list[dict]] = {}
        with engine.connect() as conn:
            for record in spec.records:
                sql = record.retrieval.get("sql")
                if not sql:
                    result[record.record_category] = []
                    continue
                rows = conn.execute(text(sql), params).mappings().all()
                result[record.record_category] = [dict(r) for r in rows]
        return result
