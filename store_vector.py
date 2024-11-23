# from langchain_experimental.text_splitter import SemanticChunker
# from langchain_openai.embeddings import OpenAIEmbeddings
import chromadb
from langchain_community.document_loaders import DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# フォルダからファイルをロードする
loader = DirectoryLoader(
    "data", glob="**/*.md", show_progress=True, use_multithreading=True
)
docs = loader.load()

# Documentオブジェクトからテキストを抽出する
doc_texts = [doc.page_content for doc in docs]

# 読み込んだ内容を分割する
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)
split_docs = text_splitter.split_documents(docs)

# ベクトル化する準備
embedding_model = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# 埋め込みベクトルを生成
embedding_vectors = embedding_model.embed_documents(
    [doc.page_content for doc in split_docs]
)

# メタデータとIDを準備
metadatas = [{"source": doc.metadata.get("source", "")} for doc in split_docs]
ids = [str(i) for i in range(len(split_docs))]

# ChromaDBクライアントの初期化
chroma_client = chromadb.PersistentClient(path="./vectorDB")
collection = chroma_client.get_or_create_collection(name="medical_record")

# データを追加
collection.add(
    embeddings=embedding_vectors,
    documents=[doc.page_content for doc in split_docs],
    metadatas=metadatas,
    ids=ids,
)

print("データの追加に成功しました。")

# chroma run --path ./vectorDB
# # サーバーモードのchromaに接続するためには、HTTPClientを利用する
# import chromadb
# client = chromadb.HttpClient(host='localhost', port=8000)
