import glob
import json
import os
import random
import re
import sys
import time
from datetime import datetime

import requests

url = "http://localhost:5000/ask"
headers = {"Content-Type": "application/json"}

# スクリプトがある場所から相対パスで指定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "1_preprocess/hanwa_kinen/input")
INSTRUCTIONS_PATH = os.path.join(BASE_DIR, "instructions_inputs.json")

# 出力ディレクトリをスクリプトからの相対パスに変更
OUTPUT_DIR = os.path.join(BASE_DIR, "2_llm")

# リクエスト間の待機時間（秒）
REQUEST_DELAY = 5

# バッチサイズ
BATCH_SIZE = 5

# 最大再試行回数
MAX_RETRIES = 3


def load_instructions(json_path):
    """JSONファイルから質問内容を読み込む"""
    # デフォルトの質問
    default_instructions = {"看護サマリ": "看護サマリを作成してください"}

    try:
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                instructions = json.load(f)

            # 読み込んだ内容が有効かチェック
            if not instructions or not isinstance(instructions, dict):
                print(
                    f"instructions_inputs.jsonの内容が無効です。デフォルトの質問を使用します。"
                )
                return default_instructions

            return instructions
        else:
            print(f"instructions_inputs.jsonが見つかりません: {json_path}")
            # デフォルトの質問を返す
            return default_instructions
    except Exception as e:
        print(f"instructions_inputs.jsonの読み込みに失敗しました: {e}")
        # デフォルトの質問を返す
        return default_instructions


def read_markdown(context_file_path):
    """Markdownファイルからテキストを読み込む"""
    if not os.path.exists(context_file_path):
        print(f"ファイルが存在しません: {context_file_path}")
        return ""
    try:
        with open(context_file_path, encoding="utf-8") as f:
            # ファイル内容全体を文字列にする
            content = f.read()
        return content
    except Exception as e:
        print(
            f"Markdownファイル {context_file_path} の読み込みに失敗しました: {e}"
        )
        return ""


def extract_context_question(file_path, instructions):
    """
    ファイルからコンテキストを読み込み、instructionsからランダムに質問を選択する
    """
    context = read_markdown(file_path)
    if not context:
        return "", ""

    if not instructions:
        # instructionsが空の場合のデフォルト
        question = "看護サマリを作成してください"
    else:
        # instructionsのvalueの中からランダムに1つ選ぶ
        question = random.choice(list(instructions.values()))

    return context, question


def get_answer(question, context, retries=MAX_RETRIES):
    """
    質問とコンテキストをAPIに送信し、回答を取得する
    エラー時に再試行メカニズムを追加
    """
    payload = {"question": question, "context": context}

    for attempt in range(retries):
        try:
            print(f"APIにリクエスト送信中... (試行 {attempt+1}/{retries})")
            response = requests.post(
                url, headers=headers, data=json.dumps(payload)
            )

            # レスポンスの詳細をログに出力
            print(f"APIレスポンスステータス: {response.status_code}")
            print(f"APIレスポンスヘッダー: {response.headers}")
            print(f"APIレスポンステキスト(生): {response.text}")

            if response.status_code == 200:
                try:
                    # APIのレスポンスから answer キーの値を取得
                    response_json = response.json()
                    answer = response_json.get("answer", "")
                    if answer:
                        print(f"回答を受信しました。長さ: {len(answer)} 文字")
                    else:
                        print("警告: APIから空の回答が返されました。")
                    return answer
                except json.JSONDecodeError:
                    print(
                        "エラー: APIレスポンスのJSONデコードに失敗しました。レスポンスはJSON形式ではありません。"
                    )
                    return ""
            else:
                print(f"APIエラー: ステータスコード {response.status_code}")
                try:
                    response_json = response.json()
                    error_message = response_json.get("error", "不明なエラー")
                except json.JSONDecodeError:
                    # JSONデコード失敗時は生のテキストをエラーメッセージとする
                    error_message = response.text

                print(f"エラーメッセージ: {error_message}")

                # GPUメモリ不足エラーの場合は待機時間を長くして再試行
                if (
                    "CUDA out of memory" in error_message
                    or "GPUメモリ不足" in error_message
                ):
                    wait_time = REQUEST_DELAY * (
                        attempt + 2
                    )  # 徐々に待機時間を増やす
                    print(
                        f"GPUメモリ不足エラー。{wait_time}秒待機してから再試行します..."
                    )
                    time.sleep(wait_time)
                    continue

                return ""
        except requests.exceptions.ConnectionError:
            print(
                "APIサーバーに接続できません。サーバーが起動しているか確認してください。"
            )
            return ""
        except Exception as e:
            print(f"予期せぬエラーが発生しました: {e}")
            return ""


def check_api_server(url="http://127.0.0.1:5001/"):
    """APIサーバーが起動しているか確認する"""
    print("APIサーバーの起動確認中...")
    try:
        test_response = requests.get(url, timeout=5)
        if test_response.status_code == 200:
            print("APIサーバーは正常に起動しています。")
            return True
        else:
            print(
                "APIサーバーに接続できますが、正しく応答していません。"
                f"ステータス: {test_response.status_code}"
            )
            return False
    except requests.exceptions.ConnectionError:
        print("APIサーバーに接続できません。")
        return False
    except Exception as e:
        print(f"APIサーバーの確認中にエラーが発生しました: {e}")
        return False


def get_processed_files(progress_file):
    """処理済みファイルのリストを取得"""
    if not os.path.exists(progress_file):
        return []

    with open(progress_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def extract_patient_id(file_path):
    """ファイル名から患者IDを抽出する"""
    # ファイル名を取得してパスと拡張子を除去
    file_name = os.path.basename(file_path)
    file_name_without_ext = os.path.splitext(file_name)[0]
    return file_name_without_ext


def ensure_directory_exists(directory):
    """ディレクトリが存在しない場合は作成する"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def write_progress_log(message):
    """進捗ログを記録する"""
    log_file = os.path.join(OUTPUT_DIR, "progress.log")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def process_data(file_suffix=""):
    # 1. 指定されたJSONファイルから質問内容を読み込む
    instructions = load_instructions(INSTRUCTIONS_PATH)
    if not instructions:
        print("質問内容が読み込めませんでした。デフォルトの質問を使用します。")
        instructions = {"看護サマリ": "看護サマリを作成してください"}

    print(f"使用する質問: {instructions}")

    # 2. 入力ディレクトリ内のすべてのMarkdownファイルを取得
    md_files = glob.glob(os.path.join(INPUT_DIR, "*.md"))
    if not md_files:
        print(
            f"指定されたディレクトリ {INPUT_DIR} にMarkdownファイルが見つかりませんでした。"
        )
        return

    print(f"処理するファイル数: {len(md_files)}")
    write_progress_log(f"処理を開始: 対象ファイル数 {len(md_files)}")

    # APIサーバーの接続確認
    try:
        # /ask を / に置換してヘルスチェックエンドポイントを叩く
        health_check_url = url.replace("/ask", "/")
        test_response = requests.get(health_check_url)
        if test_response.status_code != 200:
            print(
                f"APIサーバーに接続できますが、正しく応答していません。"
                f"ステータス: {test_response.status_code}"
            )
    except requests.exceptions.ConnectionError:
        print(
            "APIサーバーに接続できません。サーバーが起動しているか確認してください。"
        )
        print(
            "まず 'uv run python app.py' を別のコンソールで実行してください。"
        )
        if input("サーバーを起動せずに続行しますか？ (y/n): ").lower() != "y":
            return
    except Exception as e:
        print(f"APIサーバーの確認中に予期せぬエラーが発生しました: {e}")

    # 対象のファイルを限定（テスト用）
    test_mode = (
        input(
            "テストモードで実行しますか？一部のファイルのみ処理します (y/n): "
        ).lower()
        == "y"
    )
    if test_mode:
        md_files = md_files[:3]  # 最初の3ファイルのみ処理
        print(
            f"テストモード: 処理するファイル数を {len(md_files)} に制限しました"
        )
        write_progress_log(
            f"テストモード: 処理するファイル数を {len(md_files)} に制限"
        )

    # 既に処理済みのファイルリストを取得（途中から再開できるように）
    progress_file = os.path.join(OUTPUT_DIR, "processed_files.txt")
    processed_files = get_processed_files(progress_file)
    if processed_files:
        print(f"すでに {len(processed_files)} ファイルが処理済みです")
        write_progress_log(f"処理済みファイル数: {len(processed_files)}")

    # メモリ競合を軽減するため、ファイルをランダムに並べ替え
    random.shuffle(md_files)

    # 処理対象のファイルを絞り込む（処理済みのファイルを除外）
    md_files_to_process = [
        f for f in md_files if extract_patient_id(f) not in processed_files
    ]

    if not md_files_to_process:
        print("すべてのファイルの処理が完了しています。")
        return

    total_files = len(md_files_to_process)
    print(f"未処理のファイル数: {total_files}")

    # バッチ処理
    BATCH_SIZE = 5  # 一度に処理するファイル数
    total_batches = (total_files + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, total_files, BATCH_SIZE):
        batch_files = md_files_to_process[i : i + BATCH_SIZE]
        batch_count = (i // BATCH_SIZE) + 1
        print(
            f"\nバッチ {batch_count}/{total_batches} を処理中... "
            f"(ファイル数: {len(batch_files)})"
        )
        write_progress_log(
            f"バッチ {batch_count}/{total_batches} 処理開始 "
            f"(ファイル数: {len(batch_files)})"
        )

        for file_path in batch_files:
            try:
                # 拡張子なしの患者IDを使用
                patient_id = extract_patient_id(file_path)
                print(
                    f"\n処理中のファイル: {os.path.basename(file_path)} "
                    f"(ID: {patient_id})"
                )

                # 患者ごとの出力ディレクトリを作成
                patient_output_dir = os.path.join(OUTPUT_DIR, patient_id)
                ensure_directory_exists(patient_output_dir)

                # file_suffixが未定義だった問題を修正
                output_file_path = os.path.join(
                    patient_output_dir,
                    f"{patient_id}{file_suffix}_summary.txt",
                )

                # 既存のファイルがある場合はスキップ
                if os.path.exists(output_file_path):
                    try:
                        with open(
                            output_file_path, "r", encoding="utf-8"
                        ) as f:
                            content = f.read()
                        if content.strip():
                            print("既存のファイルを使用します")
                            continue
                    except Exception as e:
                        print(f"既存ファイルの読み込みに失敗: {e}")

                # コンテキストと質問を抽出
                context, question = extract_context_question(
                    file_path, instructions
                )

                # questionが空の場合のデフォルト処理
                if not question.strip():
                    question = "看護サマリを作成してください"

                print(f"  使用する質問: {question}")

                # API呼び出しで回答を取得
                answer = get_answer(question, context)

                # 回答をファイルに保存
                if answer:
                    print(f"回答の保存を試みます: {output_file_path}")
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        f.write(answer)
                    print(f"回答を保存しました: {output_file_path}")

                    # 処理済みファイルリストに追加して記録
                    if patient_id not in processed_files:
                        with open(progress_file, "a", encoding="utf-8") as f:
                            f.write(f"{patient_id}\n")
                        processed_files.append(patient_id)
                else:
                    print("回答が空のため、保存しませんでした")
                    write_progress_log(f"患者ID: {patient_id} 回答が空でした")
            except Exception as e:
                # より汎用的なエラーメッセージに変更
                print(
                    f"ファイル {os.path.basename(file_path)} "
                    f"の処理中にエラーが発生しました: {e}"
                )
                write_progress_log(
                    f"エラー: {os.path.basename(file_path)} の処理中に失敗 - {e}"
                )

        # バッチ処理の後、より長い待機時間を設定してGPUメモリを解放
        wait_time = REQUEST_DELAY * 2
        print(f"バッチ処理完了。次のバッチまで {wait_time}秒待機中...")
        write_progress_log(f"バッチ {batch_count}/{total_batches} 処理完了")
        time.sleep(wait_time)


if __name__ == "__main__":
    start_time = datetime.now()
    print(f"処理を開始します: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"入力ディレクトリ: {INPUT_DIR}")
    print(f"出力ディレクトリ: {OUTPUT_DIR}")

    # 入力ディレクトリの存在確認
    if not os.path.exists(INPUT_DIR):
        print(f"入力ディレクトリが存在しません: {INPUT_DIR}")
        sys.exit(1)

    # 出力ディレクトリの作成
    ensure_directory_exists(OUTPUT_DIR)

    # 進捗ログの初期化
    write_progress_log(f"処理開始: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    try:
        process_data()
        end_time = datetime.now()
        processing_time = end_time - start_time
        print(f"処理が完了しました: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"処理時間: {processing_time}")
        write_progress_log(
            f"処理完了: {end_time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        write_progress_log(f"処理時間: {processing_time}")
    except KeyboardInterrupt:
        print("\n処理が中断されました。")
        write_progress_log("処理が中断されました")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        write_progress_log(f"エラー発生: {e}")
        raise
