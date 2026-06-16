"""query_spec / codesystem のローダーと検証。

YAML を Pydantic で検証し、record_category が RecordCategory のメンバか、
retrieval.sql のバインド変数が許可された2種（:patient_id / :encounter_id）
のみかを起動時に検査する（SQLインジェクション・設定誤りの防止）。

詳細設計: docs/db-input-design.md §2.4, §7.3 を参照。
"""

import re
from pathlib import Path

import yaml

from adapters.models import RecordCategory
from query_specs_loader.models import QuerySpec

SPEC_DIR = Path(__file__).resolve().parent.parent / "query_specs"
CODESYSTEM_DIR = SPEC_DIR / "codesystems"

# retrieval.sql で許可するバインド変数
_ALLOWED_BIND_VARS = {"patient_id", "encounter_id"}
_BIND_VAR = re.compile(r":(\w+)")


def load_query_spec(spec_id: str) -> QuerySpec:
    """query_spec YAML をロード・検証して返す。

    Args:
        spec_id: 取得仕様ID（例: "sql_sample"）。

    Returns:
        検証済み QuerySpec。

    Raises:
        FileNotFoundError: ファイルが存在しない場合。
        ValueError: record_category が不正、または retrieval.sql に
            許可されないバインド変数が含まれる場合。
    """
    path = SPEC_DIR / f"{spec_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"query_spec '{spec_id}' が見つかりません: {path}"
        )

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    spec = QuerySpec.model_validate(data)

    for record in spec.records:
        # record_category が RecordCategory のメンバであること
        try:
            RecordCategory(record.record_category)
        except ValueError as exc:
            raise ValueError(
                f"query_spec '{spec_id}': 未知の record_category "
                f"'{record.record_category}'"
            ) from exc
        # retrieval.sql のバインド変数を検査
        sql = record.retrieval.get("sql")
        if sql:
            used = set(_BIND_VAR.findall(sql))
            illegal = used - _ALLOWED_BIND_VARS
            if illegal:
                raise ValueError(
                    f"query_spec '{spec_id}' の record "
                    f"'{record.record_category}': 許可されないバインド変数 "
                    f"{sorted(illegal)}（:patient_id / :encounter_id のみ可）"
                )

    return spec


def load_codesystem(codesystem_id: str) -> dict[str, str]:
    """コード体系 YAML をロードしコード→名称の辞書を返す。

    Args:
        codesystem_id: コード体系ID（例: "medis_obs"）。

    Returns:
        コード→名称の辞書。ファイルが無ければ空辞書。
    """
    path = CODESYSTEM_DIR / f"{codesystem_id}.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    codes = data.get("codes", {})
    return {str(k): str(v) for k, v in codes.items()}
