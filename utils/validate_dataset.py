import json
import sys

# --- ここにあなたのデータセットファイル名を入力してください ---
DATASET_FILENAME = "../data/finetuning/finetuning_data.jsonl"
# ---------------------------------------------------------


def validate_jsonl(filename):
    """
    JSONLファイルが正しい形式か検証するスクリプト。
    問題があれば、エラー内容と行番号を表示して終了する。
    """
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line_number = i + 1

                # 空行をチェック
                if not line.strip():
                    print(
                        f"❌ エラー: {line_number}行目が空です。空行は削除してください。"
                    )
                    sys.exit(1)

                # JSONとしてパースできるかチェック
                try:
                    json.loads(line)
                except json.JSONDecodeError as e:
                    print(
                        f"❌ エラー: {line_number}行目が有効なJSONではありません。"
                    )
                    print(f"   詳細: {e}")
                    print(f"   問題の行: {line.strip()}")
                    sys.exit(1)

        print(f"✅ 検証成功: '{filename}' は正しいJSONL形式です。")

    except FileNotFoundError:
        print(
            f"❌ エラー: ファイル '{filename}' が見つかりません。ファイル名やパスを確認してください。"
        )
        sys.exit(1)


if __name__ == "__main__":
    validate_jsonl(DATASET_FILENAME)
