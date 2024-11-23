import torch
from torch import cuda, bfloat16
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import pandas as pd
import chromadb
from chromadb.config import Settings
from langchain.embeddings import HuggingFaceEmbeddings
import warnings
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate


MODEL_ID = "tokyotech-llm/Swallow-70b-instruct-hf"
TOKEN = "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"

template = """
参考部分の情報を使って質問に回答してください。
### 質問
{question}
### 参考
{context}
### 回答
"""

CSV_PATH = "../instructions_inputs.csv"

warnings.simplefilter('ignore')

# パラメータを16bitから4bitに量子化
quant_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model = transformers.AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=TOKEN,
    quantization_config=quant_config,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    token=TOKEN
)

# --- RAG セットアップ ---

# ChromaDB サーバーに HTTPClient を使用して接続
chroma_client = chromadb.HttpClient(host='localhost', port=8000)

# クエリ用の埋め込みモデルを定義（同じ埋め込みモデルを使用）
embedding_function = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)
# ベクトルDBをロード
vectordb = Chroma(client=chroma_client, collection_name="medical_record", embedding_function=embedding_function)

def prepare_LLM_for_chain() -> HuggingFacePipeline:
    pipe = transformers.pipeline(
        trust_remote_code=True,
        task="text-generation",
        model=model,
        device_map="auto",
        tokenizer=tokenizer,
        max_new_tokens=512,
        repetition_penalty=1.15, #1.15
        no_repeat_ngram_size=0,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    return HuggingFacePipeline(pipeline=pipe)

if __name__=="__main__":
    llm = prepare_LLM_for_chain()
    # データを順番に処理
    data = pd.read_csv(CSV_PATH)
    for index, row in data.iterrows():
        input_text = row['input']
        prompt = PromptTemplate(template=template, input_variables=['context', 'question'])
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        docs = vectordb.similarity_search(query=input_text, k=3)
        context = "\n".join([document.page_content for document in docs])
        print("---回答---")
        print(llm_chain.run(question=input_text, context=context))

###
# chromadb.HttpClient を使用して、localhost のポート 8000 で実行されている ChromaDB サーバーに接続します。
# ChromaDB サーバーを起動するには、別のターミナルで以下を実行してください。
# chroma run --path ./vectorDB
###