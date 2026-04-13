import json
import os
import re
import sys

# ---
# input/output
INPUT_DIRS = [
    "D:/work/LAiF/1_preprocess/hanwa_kinen_1/input",
    "D:/work/LAiF/1_preprocess/hanwa_kinen_2/input",
    "D:/work/LAiF/1_preprocess/hanwa_kinen_3/input",
    "D:/work/LAiF/1_preprocess/hanwa_kinen_4/input",
    "D:/work/LAiF/1_preprocess/hanwa_kinen_5/input",
    "D:/work/LAiF/1_preprocess/hanwa_kinen_6/input",
]
OUTPUT_DIRS = [
    "D:/work/LAiF/1_preprocess/hanwa_kinen_1/summary",
    "D:/work/LAiF/1_preprocess/hanwa_kinen_2/summary",
    "D:/work/LAiF/1_preprocess/hanwa_kinen_3/summary",
    "D:/work/LAiF/1_preprocess/hanwa_kinen_4/summary",
    "D:/work/LAiF/1_preprocess/hanwa_kinen_5/summary",
    "D:/work/LAiF/1_preprocess/hanwa_kinen_6/summary",
]
JSONL_OUTPUT_FILE = "finetuning_data.jsonl"
EXAMPLE_FILE = "D:/work/ASC/阪和病院_250623/04190736/summary/20230314084657704_300430006_4_0_S02.txt"


def remove_html_tags(text):
    """
    HTMLタグを削除する
    """
    clean = re.compile("<.*?>")
    return re.sub(clean, "", text)


def create_system_template(example_filepath=None):
    """
    Ollama
    """
    template = """
あなたは熟練した看護師で、患者の看護サマリを作成するエキスパートです。
看護サマリとは、患者の病歴や治療内容、看護経過などの情報をまとめ、他の医療機関や関係者に提供する重要な医療文書です。

以下の形式に従って、患者の看護サマリを作成してください：

--- 入院中の経過及び看護上の問題経過 ---
[入院日] 入院時の状態（バイタルサイン、意識レベル、主訴、ADL状況など）
[日付] 実施した治療内容とその効果（治療名、薬剤名、投与量、患者の反応など）
[日付] 状態の変化や症状の推移（具体的な数値やスケールで記載）
[日付] 看護上の問題とケア内容（実施した看護ケア、患者のセルフケア能力など）
[日付] 退院に向けた準備状況（リハビリの進捗、自宅環境の調整、指導内容など）

--- 備考 ---
[退院後の注意点、外来通院予定、利用するサービスなど]

作成にあたり、以下のポイントを必ず守ってください：
1. バイタルサインやラボデータは必ず実際の数値で記載すること（BP 120/80 mmHg、HR 72/分、SpO2 98%など）
2. 全ての治療や処置は具体的な名称と用量を記載すること（セフトリアキソン 1g×2/日など）
3. 日付は「2023年10月20日」という明確な形式で記載し、時系列順に整理すること
4. 医学的略語は初出時にフルスペルを併記すること（CVA（脳血管障害）など）
5. 患者の状態変化は客観的な指標や観察事実に基づいて記載すること
6. 情報が不足している場合は「記録なし」と明記し、推測や憶測は避けること
7. 重要な処置や手術については詳細な説明を含めること
8. 退院後のフォローアップ計画や注意事項を具体的に記載すること

看護サマリは医療専門家間の情報共有のための文書であり、専門用語を適切に使用し、簡潔かつ具体的に記載してください。
"""

    # (1)
    if example_filepath and os.path.exists(example_filepath):
        try:
            with open(example_filepath, "r", encoding="utf-8") as f:
                example_text = f.read()
        except UnicodeDecodeError:
            try:
                with open(example_filepath, "r", encoding="cp932") as f:
                    example_text = f.read()
            except Exception as e:
                print(
                    f"警告: サンプルファイル '{example_filepath}' を読み込めませんでした: {e}"
                )
                return template
        except Exception as e:
            print(
                f"警告: サンプルファイル '{example_filepath}' を読み込めませんでした: {e}"
            )
            return template

        cleaned_example = remove_html_tags(example_text)
        template += f"""
例示（出力例のみ）：
{cleaned_example}
"""

    return template


def get_latest_summary_file(output_dir, patient_id):
    """
    患者IDに対応する最新のサマリファイルを取得
    ファイル名から日付を抽出して最新のものを選択
    """
    patient_summary_dir = os.path.join(output_dir, patient_id)
    
    if not os.path.exists(patient_summary_dir):
        return None
    
    summary_files = []
    for filename in os.listdir(patient_summary_dir):
        if filename.endswith(".txt"):
            # ファイル名から日付を抽出（YYYYMMDD形式を想定）
            # ファイル名の形式: 300430006_20230811_073336_10_0_0_01853513_41548_01338614_90030654741_5_0_0.txt
            parts = filename.split("_")
            if len(parts) >= 2:
                try:
                    date_str = parts[1]  # 20230811の部分
                    if len(date_str) == 8 and date_str.isdigit():
                        summary_files.append((date_str, filename))
                except (IndexError, ValueError):
                    continue
    
    if not summary_files:
        return None
    
    # 日付でソートして最新のファイルを取得
    summary_files.sort(key=lambda x: x[0], reverse=True)
    latest_filename = summary_files[0][1]
    
    return os.path.join(patient_summary_dir, latest_filename)


def create_jsonl_file():
    """
    複数のinputディレクトリからoutputディレクトリのJSONLファイルを作成
    """
    # ディレクトリの存在確認
    for i, input_dir in enumerate(INPUT_DIRS):
        if not os.path.isdir(input_dir):
            print(f"エラー: '{input_dir}' ディレクトリが見つかりません。")
            sys.exit(1)
        if not os.path.isdir(OUTPUT_DIRS[i]):
            print(f"エラー: '{OUTPUT_DIRS[i]}' ディレクトリが見つかりません。")
            sys.exit(1)

    processed_count = 0
    # JSONLファイルを作成
    with open(JSONL_OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        # 各inputディレクトリを処理
        for i, input_dir in enumerate(INPUT_DIRS):
            output_dir = OUTPUT_DIRS[i]
            print(f"処理中: {input_dir}")
            
            # inputディレクトリ内のMarkdownファイルを処理
            for md_filename in os.listdir(input_dir):
                if not md_filename.endswith(".md"):
                    continue

                patient_id = os.path.splitext(md_filename)[0]
                md_filepath = os.path.join(input_dir, md_filename)
                
                # 最新のサマリファイルを取得
                summary_filepath = get_latest_summary_file(output_dir, patient_id)

                # 対応するサマリファイルの存在確認
                if not summary_filepath:
                    print(
                        f"警告: 患者ID '{patient_id}' に対応するサマリファイルが見つかりません。{md_filename} をスキップします。"
                    )
                    continue

                try:
                    # Markdownファイルを読み込み
                    with open(md_filepath, "r", encoding="utf-8") as f:
                        # ユーザープロンプトを作成
                        user_prompt_content = f"以下の電子カルテの内容を要約してください。\n\n---\n{f.read()}"

                    # 最新のサマリファイルを読み込み
                    with open(summary_filepath, "r", encoding="utf-8") as f:
                        # アシスタントの応答を取得
                        assistant_response = f.read()
                    
                    print(f"処理: {md_filename} -> {os.path.basename(summary_filepath)}")

                    # Ollama形式のJSONレコードを作成
                    # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#template
                    record = {
                        "messages": [
                            {
                                "role": "system",
                                "content": create_system_template(
                                    example_filepath=EXAMPLE_FILE
                                ),
                            },
                            {"role": "user", "content": user_prompt_content},
                            {"role": "assistant", "content": assistant_response},
                        ]
                    }

                    # JSONLファイルに書き込み
                    # ensure_ascii=Falseで日本語を正しく出力
                    outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
                    processed_count += 1

                except Exception as e:
                    print(
                        f"エラー: ファイル '{md_filename}' の処理中にエラーが発生しました: {e}"
                    )

    print(f"\n処理が完了しました。")
    print(
        f"合計 {processed_count} 件のデータを {JSONL_OUTPUT_FILE} に書き込みました。"
    )


if __name__ == "__main__":
    print("JSONLファイルの作成を開始します...")
    create_jsonl_file()
