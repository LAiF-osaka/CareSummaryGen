import gc
import os
import re
import traceback
import warnings
from typing import Dict, List

import dotenv
import tiktoken
from flask import Flask, jsonify, request
from langchain_community.llms import Ollama

# .envファイルを読み込む
dotenv.load_dotenv()

# Flask アプリケーションの初期化
app = Flask(__name__)

# --- Ollama 設定 ---
MODEL_NAME = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"使用するOllamaモデル: {MODEL_NAME}")
print(f"OllamaサーバーURL: {OLLAMA_BASE_URL}")

# --- Ollama LLM のインスタンス化 ---
print("Ollama LLM を初期化中...")
try:
    # LangChain の Ollama LLM。チャット以外でも .invoke で同期推論可。
    llm = Ollama(
        model=MODEL_NAME,
        base_url=OLLAMA_BASE_URL,
        temperature=0.1,
        top_p=0.92,
        repeat_penalty=1.2,
    )
    # 接続テスト
    llm.invoke("こんにちは")
    print("Ollama LLM の初期化に成功しました。")
except Exception as e:
    print(f"Ollama の初期化または接続に失敗しました: {e}")
    raise

# --- トークナイザのロード (チャンク分割用) ---
print("Loading tokenizer for chunking (tiktoken)...")
try:
    # 汎用的で高速なトークナイザ。OpenAI 系の cl100k_base は多くの用途で安定。
    # gpt-oss:120b 固有のトークナイザではないが、チャンク分割の近似用途として十分に実用。
    # もし厳密な互換を重視する場合は、後述の TODO を参照。
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    # 万一のフォールバック
    encoding = tiktoken.get_encoding("r50k_base")

# --- Map-Reduce 定数 ---
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 130000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 1000))


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


def load_example_files():
    examples = []
    hospital_config = get_hospital_config()
    example_pairs = hospital_config["example_pairs"]

    print(f"病院設定: {hospital_config['name']}")

    for pair in example_pairs:
        try:
            with open(pair["input"], "r", encoding="utf-8") as f:
                input_text = f.read()
            with open(pair["output"], "r", encoding="utf-8") as f:
                output_text = f.read()
            examples.append({"input": input_text, "output": output_text})
        except Exception as e:
            print(f"例示ファイルの読み込みに失敗しました: {e}")
            print(f"ファイルパス: {pair['input']}, {pair['output']}")

    return examples

# --- プロンプトテンプレート (f-string形式) ---
MAP_PROMPT_FORMAT = """USER: あなたは熟練した看護師で、長い医療記録を要約する多段階プロセスの一部を担当しています。
以下に渡されるのは、医療記録の一部（チャンク）です。
このチャンクから、看護サマリの作成に必要となる全ての重要な臨床情報を時系列に沿って抽出・整理してください。

抽出するべき情報:
- 日付と時刻（必ず「2023年10月20日」のような明確な形式で記載）
- バイタルサイン（血圧、脈拍、体温、SpO2など）の具体的な数値（BP 120/80 mmHg、HR 72/分など）
- 検査データ（血液検査、画像診断など）の主要な結果と具体的な数値
- 実施された治療、処置、手術（薬剤名、投与量を含む：セフトリアキソン 1g×2/日など）
- 患者の状態変化、症状の推移（客観的な指標や観察事実に基づいて）
- 看護上の問題点と、それに対して実施されたケア（具体的な看護介入）
- ADL（日常生活動作）の変化、リハビリの進捗（具体的なスケールや評価）
- 患者や家族の重要な発言
- 医学的略語は初出時にフルスペルを併記（CVA（脳血管障害）など）

重要な注意事項:
1. 情報が不足している場合は「記録なし」と明記し、推測や憶測は避けること
2. 抽出した情報は、箇条書きや簡潔な文章でまとめること
3. この時点では、完全な文章にする必要はありません
4. 客観的事実のみを抽出し、主観的な解釈は避けること

医療記録のチャンク:
{context}
ASSISTANT:"""


def create_reduce_prompt_with_examples():
    """Few-Shot例示を含むREDUCEプロンプトを作成"""
    try:
        examples = load_example_files()
        hospital_config = get_hospital_config()

        base_prompt = f"""USER: あなたは熟練した看護師で、患者の看護サマリを作成するエキスパートです。
看護サマリとは、患者の病歴や治療内容、看護経過などの情報をまとめ、他の医療機関や関係者に提供する重要な医療文書です。

以下に、複数の医療記録チャンクから抽出・要約された臨床情報のリストが渡されます。
これらの情報を統合し、時系列に沿って整理し、一貫性のある専門的な看護サマリを以下の形式で作成してください。

以下の形式に従って、患者の看護サマリを作成してください：

{hospital_config['format_template']}

作成にあたり、以下のポイントを必ず守ってください：
1. バイタルサインやラボデータは必ず実際の数値で記載すること（BP 120/80 mmHg、HR 72/分、SpO2 98%など）
2. 全ての治療や処置は具体的な名称と用量を記載すること（セフトリアキソン 1g×2/日など）
3. 日付は「2023年10月20日」という明確な形式で記載し、時系列順に整理すること
4. 医学的略語は初出時にフルスペルを併記すること（CVA（脳血管障害）など）
5. 患者の状態変化は客観的な指標や観察事実に基づいて記載すること
6. 情報が不足している場合は「記録なし」と明記し、推測や憶測は避けること
7. 重要な処置や手術については詳細な説明を含めること
8. 退院後のフォローアップ計画や注意事項を具体的に記載すること

看護サマリは医療専門家間の情報共有のための文書であり、専門用語を適切に使用し、簡潔かつ具体的に記載してください。"""

        # 例示を追加（最大2つ）
        if examples:
            base_prompt += "\n\n# 出力例\n"
            for i, example in enumerate(examples[:2]):
                base_prompt += f"""
例示{i+1}:
{example["output"]}
"""

        base_prompt += """

抽出された臨床情報リスト:
{context}
ASSISTANT:"""

        return base_prompt
    except Exception as e:
        print(f"REDUCE プロンプトの作成に失敗しました: {e}")
        # 最小限のプロンプトを返す
        return """USER: あなたは熟練した看護師で、患者の看護サマリを作成するエキスパートです。
以下の臨床情報から看護サマリを作成してください。

抽出された臨床情報リスト:
{context}
ASSISTANT:"""


# 初期化時にプロンプトを作成
print("Few-Shot例示を含むREDUCEプロンプトを作成中...")
REDUCE_PROMPT_FORMAT = create_reduce_prompt_with_examples()
print("REDUCEプロンプトの作成完了")


# --- tiktoken ベースの分割 ---
def tokenize(text: str) -> List[int]:
    return encoding.encode(text)


def detokenize(tokens: List[int]) -> str:
    return encoding.decode(tokens)


def split_text_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """トークン数に基づくスライディングウィンドウ分割。overlap は 10–20% 推奨。"""
    toks = tokenize(text)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks: List[str] = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(toks), step):
        piece = toks[i : i + chunk_size]
        if not piece:
            break
        chunks.append(detokenize(piece))
    return chunks


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "ok",
        "message": "Ollama 看護サマリ生成APIサーバーが稼働中です",
        "model": MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
    })


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(force=True)
        context = data.get("context")
        if not context:
            return jsonify({"error": "コンテキストは必須です"}), 400

        context_tokens = tokenize(context)
        total_tokens = len(context_tokens)
        print(f"Total tokens in context: {total_tokens}")

        # コンテキストが短い場合は直接 REDUCE を実施
        if total_tokens < CHUNK_SIZE:
            print("Context is short. Generating summary directly.")
            prompt = REDUCE_PROMPT_FORMAT.format(context=context)
            answer = llm.invoke(prompt)
            return jsonify({"answer": answer})

        # --- Map-Reduceの実行 ---
        print("Context is long. Starting Map-Reduce process...")

        # 1. Mapフェーズ
        chunks = split_text_into_chunks(context, CHUNK_SIZE, CHUNK_OVERLAP)
        intermediate_summaries: List[str] = []
        print(f"Split into {len(chunks)} chunks. Starting map phase...")

        for i, chunk in enumerate(chunks):
            print(f"  Processing chunk {i+1}/{len(chunks)}...")
            map_prompt = MAP_PROMPT_FORMAT.format(context=chunk)
            summary = llm.invoke(map_prompt)
            if summary:
                intermediate_summaries.append(summary)
                print(
                    f"  Finished chunk {i+1}/{len(chunks)}. Summary length: {len(summary)}"
                )
            else:
                print(
                    f"  Warning: Chunk {i+1}/{len(chunks)} returned an empty summary."
                )

        if not intermediate_summaries:
            return jsonify({"error": "Map phase failed to produce any summaries."}), 500

        # 2. Reduceフェーズ
        print("Map phase complete. Starting reduce phase...")
        combined_context = "\n\n---\n\n".join(intermediate_summaries)
        print(f"Combined context length for reduce phase (chars): {len(combined_context)}")

        reduce_prompt = REDUCE_PROMPT_FORMAT.format(context=combined_context)
        final_answer = llm.invoke(reduce_prompt)

        print("Reduce phase complete.")
        return jsonify({"answer": final_answer})

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"エラーが発生しました: {e}\n{error_trace}")
        return jsonify({"error": f"エラーが発生しました: {e}", "trace": error_trace}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
