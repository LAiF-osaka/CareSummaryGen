import json
import os
import time

import requests
from requests.exceptions import RequestException

# --- 設定 ---
# app_ollama.py または app_custom.py のエンドポイントを指定
API_URL = "http://localhost:5000/ask"

# 入力・出力ディレクトリを指定
INPUT_DIR = "D:/work/LAiF/1_preprocess/gs_shinkinen/input"
OUTPUT_DIR = "D:/work/LAiF/2_output/gs_shinkinen/gpt-oss-120b"
# INPUT_DIR = "D:/work/LAiF/1_preprocess/gs_hanwa/input"
# OUTPUT_DIR = "D:/work/LAiF/2_output/gs_hanwa/bakeneko_original"

# リクエスト間の待機時間（秒）
SLEEP_TIME = 2
# リトライ設定
MAX_RETRIES = 3
RETRY_DELAY = 5


def send_request(patient_id: str, context: str):
    """APIにリクエストを送信し、サマリを取得する"""
    payload = {"context": context}  # questionを削除
    print(f"患者ID: {patient_id} のサマリ生成をリクエストします...")

    for attempt in range(MAX_RETRIES):
        try:
            # タイムアウトを30分に延長
            response = requests.post(API_URL, json=payload, timeout=1800)
            response.raise_for_status()  # HTTPエラーがあれば例外を発生

            result = response.json()
            # answerが空でないか、エラーが含まれていないかを確認
            if "answer" in result and result["answer"]:
                print(f"患者ID: {patient_id} のサマリ生成に成功しました。")
                return result["answer"]
            else:
                error_message = result.get(
                    "error", "不明なエラーまたは空のレスポンス"
                )
                print(f"APIエラー: {error_message}")
                return None

        except RequestException as e:
            print(
                f"リクエスト中にエラーが発生しました (試行 {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            if attempt < MAX_RETRIES - 1:
                print(f"{RETRY_DELAY}秒後に再試行します...")
                time.sleep(RETRY_DELAY)
            else:
                print(
                    "最大再試行回数に達しました。この患者IDの処理をスキップします。"
                )
                return None


def process_patient_files():
    """指定されたディレクトリを処理し、サマリを生成・保存する"""
    if not os.path.exists(INPUT_DIR):
        print(f"エラー: 入力ディレクトリが見つかりません: {INPUT_DIR}")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"入力ディレクトリ: {os.path.abspath(INPUT_DIR)}")
    print(f"出力ディレクトリ: {os.path.abspath(OUTPUT_DIR)}")

    # .mdファイルを直接探す
    md_files = [
        f
        for f in os.listdir(INPUT_DIR)
        if f.endswith('.md') and os.path.isfile(os.path.join(INPUT_DIR, f))
    ]
    
    # ファイル名から患者IDを抽出（拡張子を除く）
    patient_ids = [os.path.splitext(f)[0] for f in md_files]
    
    total_patients = len(patient_ids)
    print(f"合計 {total_patients} 人の患者データを処理します。")

    for i, patient_id in enumerate(patient_ids):
        print(
            f"--- 患者 {i + 1}/{total_patients} (ID: {patient_id}) の処理を開始 ---"
        )
        patient_output_dir = os.path.join(OUTPUT_DIR, patient_id)

        # 出力ファイルが既に存在する場合はスキップ
        output_file_path = os.path.join(patient_output_dir, "summary.txt")
        if os.path.exists(output_file_path):
            print(
                f"出力ファイルが既に存在するため、スキップします: {output_file_path}"
            )
            continue

        # .md ファイルを読み込む
        context = ""
        try:
            md_file_path = os.path.join(INPUT_DIR, f"{patient_id}.md")
            if not os.path.exists(md_file_path):
                print(
                    f"警告: ファイルが見つかりません: {md_file_path}。スキップします。"
                )
                continue

            print(
                f"入力ファイルを読み込みます: {os.path.abspath(md_file_path)}"
            )
            with open(md_file_path, "r", encoding="utf-8") as f:
                context = f.read()

            print(f"コンテキストを読み込みました (合計 {len(context)} 文字)。")

        except Exception as e:
            print(
                f"ファイル読み込み中にエラーが発生しました (患者ID: {patient_id}): {e}"
            )
            continue

        # APIリクエストと結果の保存
        summary = send_request(patient_id, context)

        if summary:
            try:
                os.makedirs(patient_output_dir, exist_ok=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"サマリを保存しました: {output_file_path}")
            except Exception as e:
                print(f"ファイル書き込み中にエラーが発生しました: {e}")

        # サーバーへの負荷軽減
        print(f"{SLEEP_TIME}秒待機します...")
        time.sleep(SLEEP_TIME)

    print("--- すべての処理が完了しました ---")


if __name__ == "__main__":
    process_patient_files()
