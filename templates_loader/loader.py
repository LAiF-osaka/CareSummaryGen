"""テンプレート管理モジュール。

YAML ファイルから機関別の看護サマリーテンプレートをロードする。
テンプレートはセクション定義・区切り書式・Few-Shot 例示パスを持ち、
コード変更なしで新機関を追加できる。
"""

from pathlib import Path

import yaml

# テンプレートディレクトリ（プロジェクトルート/templates/）
TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def load_template(template_id: str) -> dict:
    """テンプレート ID に対応する YAML ファイルをロードする。

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
    """利用可能なテンプレートの一覧を返す。

    Returns:
        テンプレート情報（id, name, description）のリスト。
    """
    templates = []
    for path in sorted(TEMPLATE_DIR.glob("*.yaml")):
        # routing 定義（<id>.routing.yaml）はテンプレート本体ではない
        if path.name.endswith(".routing.yaml"):
            continue
        with open(path, encoding="utf-8") as f:
            t = yaml.safe_load(f)
            templates.append(
                {
                    "id": t["id"],
                    "name": t["name"],
                    "description": t.get("description", ""),
                }
            )
    return templates


def build_format_instruction(template: dict) -> str:
    """テンプレートから LLM へのフォーマット指示文を構築する。

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
