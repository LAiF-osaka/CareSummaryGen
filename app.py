import gc
import os
import re
import traceback
import warnings
from typing import Dict, List

import dotenv
import torch
import transformers
from flask import Flask, jsonify, request
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from pydantic import BaseModel, Field
from torch import bfloat16, cuda
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# .envファイルを読み込む
dotenv.load_dotenv()

# GPU設定の環境変数を取得
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES")
PYTORCH_CUDA_ALLOC_CONF = os.environ.get(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF

# 利用可能なGPUの確認
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
    print(f"利用可能なGPU: {gpu_count}台")
    for i, name in enumerate(gpu_names):
        print(f"  GPU {i}: {name}")

    # 現在設定されているGPU
    if CUDA_VISIBLE_DEVICES:
        print(f"使用するGPU設定: CUDA_VISIBLE_DEVICES={CUDA_VISIBLE_DEVICES}")
    else:
        print("すべてのGPUが利用可能です")
else:
    print("使用可能なGPUがありません")

# メモリ最適化の設定
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Flask アプリケーションの初期化
app = Flask(__name__)

# MODEL_ID = "tokyotech-llm/Llama-3.3-Swallow-70B-Instruct-v0.4"
# MODEL_ID = "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8"
# MODEL_ID = "tokyotech-llm/Llama-3.1-Swallow-8B-Instruct-v0.3"
# MODEL_ID = "JPishikawa/Llama-3.3-Swallow-70B-Instruct-v0.4-FP8-Dynamic"
# MODEL_ID = "rinna/qwq-bakeneko-32b"
MODEL_ID = "rinna/qwen2.5-bakeneko-32b-instruct-v2"
TOKEN = os.environ.get("HF_TOKEN", "")

# CPU実行モードフラグ
USE_CPU = False


# 指定されたGPUデバイスを使用
def select_gpu(gpu_index=None):
    if not torch.cuda.is_available() or USE_CPU:
        return "cpu"

    if gpu_index is not None and 0 <= gpu_index < torch.cuda.device_count():
        return f"cuda:{gpu_index}"
    else:
        return "cuda"


# 環境変数からGPUインデックスを取得（指定がなければNone）
GPU_INDEX = None
if CUDA_VISIBLE_DEVICES and len(CUDA_VISIBLE_DEVICES.split(",")) == 1:
    try:
        GPU_INDEX = int(CUDA_VISIBLE_DEVICES)
        print(f"環境変数から指定されたGPUインデックス: {GPU_INDEX}")
    except ValueError:
        print("環境変数CUDA_VISIBLE_DEVICESの値が正しくありません")


# GPUメモリ情報を表示
def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            free_gb = free_mem / (1024**3)
            total_gb = total_mem / (1024**3)
            used_gb = total_gb - free_gb
            print(
                f"GPU {i} メモリ: 使用={used_gb:.2f}GB / 合計={total_gb:.2f}GB (空き={free_gb:.2f}GB)"
            )


# モデルロード用のパラメータ設定
# quant_config = transformers.BitsAndBytesConfig(load_in_8bit=True)

# グローバル変数でモデルと関連リソースを保持し、一度ロードしたら再利用
print("Loading model...")
try:
    # GPUメモリのキャッシュをクリア
    if torch.cuda.is_available() and not USE_CPU:
        torch.cuda.empty_cache()
        gc.collect()
        print_gpu_memory()

    device = select_gpu(GPU_INDEX)
    device_map = device if device == "cpu" else "auto"

    print(f"モデルを {device} でロードします")

    # attn_implementation を追加 (環境に合わせて "sdpa" または "flash_attention_2" を試す)
    # PyTorch 2.0 以降が必要
    attn_impl = "sdpa" if torch.__version__ >= "2.0" else None
    if attn_impl:
        print(f"Attention implementation: {attn_impl} を使用します")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=TOKEN,
        # quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
        device_map=device_map,
        attn_implementation=attn_impl,  # attn_implementation を追加
    )

    print(f"モデルのロード成功: {'CPU' if device == 'cpu' else 'GPU'}")
    if torch.cuda.is_available():
        print_gpu_memory()

except Exception as e:
    print(f"GPUでのモデルロード中にエラーが発生: {e}")
    print("CPUでモデルをロードします...")
    USE_CPU = True
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        token=TOKEN,
        torch_dtype=torch.float32,
        device_map="cpu",
    )
    print("CPUでモデルをロードしました")

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)

print("Setting up pipeline...")
pipe = pipeline(
    trust_remote_code=True,
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    repetition_penalty=1.3,  # 1.3で調整
    # no_repeat_ngram_size=4, # n_gramの繰り返しの防止については不要
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    temperature=0.1,
    top_p=0.92,  # 出力の多様性を適度に制限（0.9から0.92に変更）
    return_full_text=False,
)

llm = HuggingFacePipeline(pipeline=pipe)


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


def create_system_template():
    try:
        examples = load_example_files()
        hospital_config = get_hospital_config()

        template = f"""
あなたは熟練した看護師で、患者の看護サマリを作成するエキスパートです。
看護サマリとは、患者の病歴や治療内容、看護経過などの情報をまとめ、他の医療機関や関係者に提供する重要な医療文書です。

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

看護サマリは医療専門家間の情報共有のための文書であり、専門用語を適切に使用し、簡潔かつ具体的に記載してください。
"""

        # 例示を追加（2つの例示を使用）
        if examples:
            template += "\n例示（出力例）：\n"
            for i, example in enumerate(examples[:2]):  # 最大2つの例示を使用
                template += f"""
例示{i+1}:
{example["output"]}
"""

        return template
    except Exception as e:
        print(f"システムテンプレートの作成に失敗しました: {e}")
        # 最小限のテンプレートを返す
        return """
あなたは熟練した看護師で、患者の看護サマリを作成するエキスパートです。
看護サマリとは、患者の病歴や治療内容、看護経過などの情報をまとめ、他の医療機関や関係者に提供する重要な医療文書です。

以下の形式で看護サマリを作成してください：

--- 入院中の経過及び看護上の問題経過 ---
[入院日] 入院時の状態
[日付] 主な治療内容
[日付] 状態の変化

--- 備考 ---
[備考事項]

特に以下の点に注意してください：
1. 日付は必ず「2023年10月20日」のような明確な形式で記載する
2. 客観的事実に基づいて記載する
3. 情報が不足している場合は推測せず、「記録なし」と記載する
"""


# システムプロンプトの初期化
try:
    system_template = create_system_template()
    print("システムテンプレートを作成しました")
except Exception as e:
    print(f"システムテンプレートの初期化に失敗しました: {e}")
    system_template = """
あなたは病院で働く看護師です。看護サマリを作成してください。
"""

# 以下は既存のコードと同じ
user_template = """
与えられた医療記録から看護サマリを作成してください。

医療記録:
{context}
"""

# 元のプロンプト
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_template,
        ),
        ("user", user_template),
    ]
)


class Summary(BaseModel):
    summary: str = Field(description="看護サマリ内容")


output_parser = (
    StrOutputParser()
)  # PydanticOutputParser(pydantic_object=Summary)
# format_instructions = output_parser.get_format_instructions()


def split_chunks(context: str, window_size=500, overlap=250) -> List[str]:
    """
    コンテキストを重複ありのチャンクに分割する
    チャンクサイズとオーバーラップを調整して、重要な文脈が途切れないようにする
    """
    chunks = []
    lines = context.split("\n")
    current_chunk = []
    current_length = 0

    for line in lines:
        # 行が非常に長い場合は適切に分割
        if len(line) > window_size * 2:
            # 長い行は単語単位で分割
            words = line.split()
            temp_line = ""
            for word in words:
                if len(temp_line) + len(word) + 1 <= window_size * 2:
                    temp_line += word + " "
                else:
                    chunks.append(temp_line.strip())
                    temp_line = word + " "
            if temp_line:
                chunks.append(temp_line.strip())
            continue

        # 通常の行処理
        if current_length + len(line) <= window_size:
            current_chunk.append(line)
            current_length += len(line)
        else:
            # チャンクが最小サイズ未満なら追加の行を含める
            if current_length < 100 and lines:
                additional_lines = min(3, len(lines) - lines.index(line))
                for _ in range(additional_lines):
                    if lines and lines[0]:
                        current_chunk.append(lines.pop(0))

            # 現在のチャンクを追加
            chunks.append("\n".join(current_chunk))

            # 重要な文脈を維持するためのオーバーラップ
            overlap_lines = current_chunk[
                -min(
                    len(current_chunk),
                    int(len(current_chunk) * overlap / window_size),
                ) :
            ]

            # 新しいチャンクを開始（オーバーラップ含む）
            current_chunk = overlap_lines + [line]
            current_length = sum(len(l) for l in current_chunk)

    # 残りのチャンクを追加
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    # 各チャンクに位置情報を追加
    for i in range(len(chunks)):
        position_info = f"[チャンク {i+1}/{len(chunks)}]\n"
        chunks[i] = position_info + chunks[i]

    return chunks


def execute_chain(context: str) -> str:
    """
    LLMチェーンを実行し、看護サマリを生成する
    シンプルな処理方法に変更
    """
    try:
        # 入力の前処理
        context = context.strip()
        llm_chain = prompt | llm | output_parser

        # GPUメモリをクリア
        if torch.cuda.is_available() and not USE_CPU:
            print("実行前のGPUメモリ状態:")
            print_gpu_memory()
            torch.cuda.empty_cache()
            gc.collect()

        # コンテキストが長すぎる場合は制限
        if len(context) > 3000:
            print(
                f"コンテキストが長いため、最初の3000文字に制限します。元の長さ: {len(context)}文字"
            )
            context = context[:3000]

        # プロンプトを作成
        full_prompt = prompt.format(context=context)
        print(f"プロンプト:\n{full_prompt[:500]}... (省略)")

        # 単一のLLM呼び出しでサマリを生成
        try:
            answer = llm_chain.invoke(
                input={
                    "context": context,
                }
            )
            print(f"サマリー生成完了: {len(answer)} 文字")
            print(answer)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"GPUメモリ不足エラーが発生しました。処理を中止します。")
                return "GPUメモリ不足エラーが発生しました。処理を中止します。"
            else:
                raise e

        # 出力の検証
        if not answer:
            return "看護サマリを生成できませんでした。入力データを確認してください。"

        return answer
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"execute_chainでエラーが発生しました: {e}")
        print(f"詳細: {error_trace}")
        return f"エラーが発生しました: {str(e)}"


@app.route("/", methods=["GET"])
def index():
    """ルートパスへのアクセスに対するレスポンス"""
    return jsonify(
        {"status": "ok", "message": "看護サマリ生成APIサーバーが稼働中です"}
    )


@app.route("/ask", methods=["POST"])
def ask():
    try:
        # リクエストからコンテキストを取得
        data = request.get_json()
        context = data.get("context", "")

        print(f"リクエスト受信: コンテキスト長={len(context)}")

        # コンテキストの内容をログ出力
        print(
            f"コンテキスト: {context[:500]}..."
            if len(context) > 500
            else f"コンテキスト: {context}"
        )

        # 入力の検証
        if not context:
            print("コンテキストが空です")
            return jsonify({"error": "コンテキストは必須です"}), 400

        if len(context) > 100000:  # 適切な上限を設定
            print(f"コンテキストが長すぎます: {len(context)} 文字")
            return jsonify({"error": "Context too long."}), 400

        try:
            # リクエスト処理前にGPUメモリをクリア
            if torch.cuda.is_available() and not USE_CPU:
                print("GPUメモリをクリアします...")
                torch.cuda.empty_cache()
                gc.collect()
                print("クリア後のGPUメモリ状態:")
                print_gpu_memory()

            # LLMチェーンを実行
            print("LLMチェーンを実行中...")
            answer = execute_chain(context)

            # GPUメモリ不足エラーのチェック
            if answer and "GPUメモリ不足エラー" in answer:
                print("GPUメモリ不足エラーが発生しました。")
                return (
                    jsonify({"error": answer}),
                    503,
                )  # 503 Service Unavailable

            # 出力の検証
            if not answer or len(answer) < 10:
                print(
                    f"生成されたサマリーが短すぎます: {len(answer) if answer else 0} 文字"
                )
                return (
                    jsonify(
                        {"error": "Generated summary too short or empty."}
                    ),
                    500,
                )

            print(f"サマリー生成完了: {len(answer)} 文字")
            print(f"生成したサマリー: {answer[:200]}...")

            # 結果を返す
            response = jsonify({"answer": answer})
            return response

        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"execute_chainでエラーが発生しました: {e}")
            print(f"詳細: {error_trace}")

            # CUDA OOMエラーの場合、シンプルなエラーメッセージを返す
            if "CUDA out of memory" in str(e):
                return (
                    jsonify(
                        {
                            "error": "GPUメモリ不足エラーが発生しました。しばらく待ってから再試行してください。"
                        }
                    ),
                    503,
                )  # 503 Service Unavailable

            return (
                jsonify(
                    {
                        "error": f"Failed to generate summary: {str(e)}",
                        "details": error_trace,
                    }
                ),
                500,
            )

    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"askエンドポイントでエラーが発生しました: {e}")
        print(f"詳細: {error_trace}")
        return jsonify({"error": str(e), "details": error_trace}), 500


if __name__ == "__main__":
    # Flask サーバーの起動
    print("看護サマリ生成APIサーバーを起動します...")
    app.run(host="0.0.0.0", port=5001, use_reloader=True)
