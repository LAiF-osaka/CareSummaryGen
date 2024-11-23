import warnings

import chromadb
import torch
import transformers
from flask import Flask, jsonify, request
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from torch import bfloat16, cuda
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Flask アプリケーションの初期化
app = Flask(__name__)

MODEL_ID = "tokyotech-llm/Swallow-70b-instruct-hf"
TOKEN = "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"

# モデルロード用のパラメータ設定
quant_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16,
)

# グローバル変数でモデルと関連リソースを保持し、一度ロードしたら再利用
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=TOKEN,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=TOKEN)

print("Setting up pipeline...")
pipe = pipeline(
    trust_remote_code=True,
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    repetition_penalty=1.15,
    no_repeat_ngram_size=0,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    temperature=0.01,
)

llm = HuggingFacePipeline(pipeline=pipe)

# ChromaDB のセットアップ
print("Connecting to ChromaDB...")
chroma_client = chromadb.HttpClient(host="localhost", port=8000)
embedding_function = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)
vectordb = Chroma(
    client=chroma_client,
    collection_name="medical_record",
    embedding_function=embedding_function,
)

# テンプレート定義
template = """
参考部分の情報を使って質問に回答してください。
### 質問
{question}
### 参考
{context}
### 回答
"""
prompt = PromptTemplate(
    template=template, input_variables=["context", "question"]
)


@app.route("/ask", methods=["POST"])
def ask():
    try:
        # リクエストから質問を取得
        data = request.get_json()
        question = data.get("question", "")
        if not question:
            return jsonify({"error": "Question is required."}), 400

        # ベクトルDBを用いた類似検索
        docs = vectordb.similarity_search(query=question, k=3)
        context = "\n".join([document.page_content for document in docs])

        # LLM チェーンの実行
        llm_chain = prompt | llm
        answer = llm_chain.invoke(
            input={"question": question, "context": context}
        )

        # 結果を返す
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Flask サーバーの起動
    app.run(host="0.0.0.0", port=5000, use_reloader=False)
