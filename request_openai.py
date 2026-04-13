import os
import time

import dotenv
import tiktoken
from openai import OpenAI

# .envファイルから環境変数を読み込む
dotenv.load_dotenv()

# --- 設定 ---
INPUT_DIR = "1_preprocess/gs_hanwa/input"
OUTPUT_DIR = "2_output/gs_hanwa/gpt-4o"

# OpenAI API設定
MODEL_NAME = "gpt-4o"

# リクエスト間の待機時間（秒）
SLEEP_TIME = 60
# リトライ設定
MAX_RETRIES = 3
RETRY_DELAY = 60

# --- OpenAI クライアントとトークナイザの初期化 ---
try:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    tokenizer = tiktoken.encoding_for_model(MODEL_NAME)
    print("OpenAI client and tokenizer initialized successfully.")
except Exception as e:
    print(f"OpenAIの初期化に失敗しました: {e}")
    exit(1)

def get_hospital_config():
    """環境変数から病院設定を取得"""
    hospital = os.environ.get("HOSPITAL", "hanwa").lower()
    
    if hospital == "shinkinen":
        return {
            "name": "新記念病院",
            "example_pairs": [
                {
                    "input": "D:/work/LAiF/1_preprocess/gs_shinkinen/example/input/20373676.md",
                    "output": "D:/work/LAiF/1_preprocess/gs_shinkinen/example/summary/20373676/20250517174222917_300430006_1_0.txt",
                },
            ],
            "format_template": """--- 入院中の経過及び看護上の問題経過 ---
[入院日] 入院時の状態（バイタルサイン、意識レベル、主訴、ADL状況など）
[日付] 実施した治療内容とその効果（治療名、薬剤名、投与量、患者の反応など）
[日付] 状態の変化や症状の推移（具体的な数値やスケールで記載）
[日付] 看護上の問題とケア内容（実施した看護ケア、患者のセルフケア能力など）
[日付] 退院に向けた準備状況（リハビリの進捗、自宅環境の調整、指導内容など）

--- 備考 ---
[退院後の注意点、外来通院予定、利用するサービスなど]"""
        }
    else:  # デフォルトは阪和病院
        return {
            "name": "阪和病院",
            "example_pairs": [
                {
                    "input": "D:/work/LAiF/1_preprocess/gs_hanwa/example/input/04190736.md",
                    "output": "D:/work/LAiF/1_preprocess/gs_hanwa/example/summary/04190736/20230314084657704_300430006_4_0_S02.txt",
                },
            ],
            "format_template": """--- 指導した内容 ---
[指導内容の詳細]

--- 医療機器装着・挿入・処置部位 ---
[医療機器や処置に関する情報]

--- 入院中の看護の経過（生活状況） ---
[日付] 生活状況や看護経過の詳細

--- 患者への病状説明及び本人・家族の受け止め方 ---
[病状説明の内容と患者・家族の反応]

--- 継続される問題（今後のリスク） ---
[継続的な問題点やリスク要因]

--- その他 ---
[その他の重要事項]"""
        }


def create_system_prompt():
    """病院設定に基づいてシステムプロンプトを作成"""
    hospital_config = get_hospital_config()
    
    return f"""あなたは熟練した看護師で、患者の看護サマリを作成するエキスパートです。
以下に、ある患者の医療記録の全文が渡されます。
これまでの対話例を参考に、情報を統合し、時系列に沿って整理し、一貫性のある専門的な看護サマリを以下の形式で作成してください。

# 注意事項
- 与えられた情報にない内容は絶対に記載せず、推測は避けること。
- 箇条書きや表ではなく、連続した文章で記載すること。
- 日付は「YYYY年M月D日」のような明確な形式で記載し、時系列順に整理すること。

# 形式
{hospital_config['format_template']}"""


# --- プロンプトテンプレート ---
SYSTEM_PROMPT = create_system_prompt()


def load_few_shot_examples():
    """病院設定に基づいてFew-shotプロンプト用の模範解答例を読み込む"""
    examples = []
    hospital_config = get_hospital_config()
    example_pairs = hospital_config["example_pairs"]
    
    print(f"病院設定: {hospital_config['name']}")
    print("Loading few-shot examples...")
    
    for pair in example_pairs:
        try:
            input_path = pair["input"]
            output_path = pair["output"]
            
            if os.path.exists(input_path) and os.path.exists(output_path):
                with open(input_path, "r", encoding="utf-8") as f:
                    input_content = f.read()
                with open(output_path, "r", encoding="utf-8") as f:
                    output_content = f.read()
                
                examples.append({"role": "user", "content": input_content})
                examples.append({"role": "assistant", "content": output_content})
                print(f"  - Loaded example: {os.path.basename(input_path)}")
            else:
                print(f"  - Warning: Example files not found. Skipping.")
                print(f"    - Checked input: {input_path}")
                print(f"    - Checked output: {output_path}")
        except Exception as e:
            print(f"  - Warning: Failed to load example. Error: {e}")
    
    return examples


# --- グローバル変数としてFew-shotプロンプトをロード ---
few_shot_examples = load_few_shot_examples()


def split_text_into_chunks(text: str, max_tokens: int) -> list[str]:
    """テキストを最大トークン数に基づいてチャンクに分割する"""
    # シンプルな改行ベースの分割
    lines = text.split("\n")
    chunks = []
    current_chunk = ""
    current_tokens = 0

    for line in lines:
        line_tokens = len(tokenizer.encode(line))
        if current_tokens + line_tokens > max_tokens:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
            current_tokens = line_tokens
        else:
            current_chunk += "\n" + line
            current_tokens += line_tokens

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def get_token_count(messages: list) -> int:
    """メッセージリストの合計トークン数を計算する"""
    count = 0
    for message in messages:
        count += len(tokenizer.encode(message["content"]))
    return count


def generate_summary_with_openai(patient_id: str, context: str) -> str:
    """OpenAI APIを直接呼び出してサマリを生成する (Map-Reduce アプローチ)"""
    print(f"患者ID: {patient_id} のサマリ生成をリクエストします...")

    # --- 設定 ---
    # GPT-4oのTPMは30,000。安全マージンを考慮し、リクエストサイズを28000トークンに制限
    MAX_REQUEST_TOKENS = 28000
    # Mapフェーズでの各チャンクの最大トークン数
    MAX_CHUNK_TOKENS = 8000

    # --- 固定プロンプトのトークン数を計算 ---
    base_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + few_shot_examples
    base_token_count = get_token_count(base_messages)

    # ユーザーコンテキストに利用可能な最大トークン数
    max_context_tokens = MAX_REQUEST_TOKENS - base_token_count
    if max_context_tokens <= 0:
        print(
            "エラー: Few-shotの例文とシステムプロンプトだけで最大トークン数を超えています。プロンプトを短くしてください。"
        )
        return None

    # --- 初期コンテキストをチャンクに分割 ---
    context_chunks = split_text_into_chunks(context, MAX_CHUNK_TOKENS)
    print(f"コンテキストを {len(context_chunks)} 個のチャンクに分割しました。")

    if not context_chunks:
        print("コンテキストが空のため、処理をスキップします。")
        return None

    # --- Mapフェーズ: 各チャンクの要約 ---
    summaries = []
    if len(context_chunks) > 1:
        for i, chunk in enumerate(context_chunks):
            print(f"  - チャンク {i+1}/{len(context_chunks)} の要約を生成中...")
            map_prompt = f"""以下は患者の医療記録の一部です。この部分の要点を時系列に沿ってまとめてください。

{chunk}"""
            messages = [
                {
                    "role": "system",
                    "content": "あなたは医療記録を要約するアシスタントです。",
                },
                {"role": "user", "content": map_prompt},
            ]
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME, messages=messages, temperature=0.1, top_p=0.9
                    )
                    summary = response.choices[0].message.content.strip()
                    if summary:
                        summaries.append(summary)
                        print(f"  - チャンク {i+1} の要約が完了しました。")
                        break
                except Exception as e:
                    print(
                        f"    - APIリクエストエラー (試行 {attempt + 1}/{MAX_RETRIES}): {e}"
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                    else:
                        print(
                            f"    - チャンク {i+1} の処理に失敗しました。スキップします。"
                        )
            time.sleep(SLEEP_TIME)

        if not summaries:
            print("中間要約を一つも生成できませんでした。処理を中止します。")
            return None
        current_context = "\n\n---\n\n".join(summaries)
    else:
        current_context = context_chunks[0]

    # --- 再帰的Reduceフェーズ ---
    while (
        get_token_count([{"role": "user", "content": current_context}])
        > max_context_tokens
    ):
        print("結合されたサマリがまだ長すぎます。さらに要約を繰り返します...")
        print(
            f"現在のトークン数: {get_token_count([{'role': 'user', 'content': current_context}])} / 許容: {max_context_tokens}"
        )

        new_chunks = split_text_into_chunks(current_context, MAX_CHUNK_TOKENS)
        summaries = []
        for i, chunk in enumerate(new_chunks):
            print(f"  - 再帰チャンク {i+1}/{len(new_chunks)} の要約を生成中...")
            map_prompt = f"""以下の複数の要約をさらに統合し、要点をまとめてください。

{chunk}"""
            messages = [
                {
                    "role": "system",
                    "content": "あなたは複数の要約を統合するアシスタントです。",
                },
                {"role": "user", "content": map_prompt},
            ]
            for attempt in range(MAX_RETRIES):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME, messages=messages, temperature=0.1, top_p=0.9
                    )
                    summary = response.choices[0].message.content.strip()
                    if summary:
                        summaries.append(summary)
                        print(f"  - 再帰チャンク {i+1} の要約が完了しました。")
                        break
                except Exception as e:
                    print(
                        f"    - APIリクエストエラー (試行 {attempt + 1}/{MAX_RETRIES}): {e}"
                    )
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
            time.sleep(SLEEP_TIME)

        if not summaries:
            print("再帰的な要約に失敗しました。処理を中止します。")
            return None
        current_context = "\n\n---\n\n".join(summaries)

    # --- 最終Reduceフェーズ ---
    print("最終的なサマリを生成します...")
    reduce_prompt = f"""以下の情報を統合し、指示された形式に従って、一貫性のある最終的な看護サマリを作成してください。

{current_context}"""
    final_messages = base_messages + [{"role": "user", "content": reduce_prompt}]

    print(f"最終リクエストの合計トークン数: {get_token_count(final_messages)}")

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME, messages=final_messages, temperature=0.1, top_p=0.9
            )
            final_summary = response.choices[0].message.content.strip()
            if final_summary:
                print(f"患者ID: {patient_id} の最終サマリ生成に成功しました。")
                return final_summary
        except Exception as e:
            print(
                f"最終サマリ生成中にAPIエラーが発生しました (試行 {attempt + 1}/{MAX_RETRIES}): {e}"
            )
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                print("最大再試行回数に達しました。")
                return None


def process_patient_files():
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
        print(f"--- 患者 {i + 1}/{total_patients} (ID: {patient_id}) の処理を開始 ---")
        patient_output_dir = os.path.join(OUTPUT_DIR, patient_id)

        output_file_path = os.path.join(patient_output_dir, "summary.txt")
        if os.path.exists(output_file_path):
            print(f"出力ファイルが既に存在するため、スキップします: {output_file_path}")
            continue

        context = ""
        try:
            md_file_path = os.path.join(INPUT_DIR, f"{patient_id}.md")
            if not os.path.exists(md_file_path):
                print(
                    f"警告: ファイルが見つかりません: {md_file_path}。スキップします。"
                )
                continue

            print(f"入力ファイルを読み込みます: {os.path.abspath(md_file_path)}")
            with open(md_file_path, "r", encoding="utf-8") as f:
                context = f.read()

            print(f"コンテキストを読み込みました (合計 {len(context)} 文字)。")

        except Exception as e:
            print(
                f"ファイル読み込み中にエラーが発生しました (患者ID: {patient_id}): {e}"
            )
            continue

        summary = generate_summary_with_openai(patient_id, context)

        if summary:
            try:
                os.makedirs(patient_output_dir, exist_ok=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    f.write(summary)
                print(f"サマリを保存しました: {output_file_path}")
            except Exception as e:
                print(f"ファイル書き込み中にエラーが発生しました: {e}")

        time.sleep(SLEEP_TIME)

    print("--- すべての処理が完了しました ---")


if __name__ == "__main__":
    process_patient_files()
