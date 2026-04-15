"""アプリケーション設定モジュール。

環境変数から設定を読み込み、アプリケーション全体で使用する定数を提供する。
LLM パラメータ・チャンク分割・Reflection の上限などを一元管理する。

環境切替:
    ENV=production (デフォルト): ローカル Ollama + gpt-oss:120b（閉域ネットワーク）
    ENV=test: Ollama Cloud + gpt-oss:120b-cloud（インターネット経由）
"""

import os

import dotenv

dotenv.load_dotenv()

# --- 環境識別 ---
ENV: str = os.environ.get("ENV", "production")

# --- Ollama 設定（環境依存） ---
if ENV == "test":
    # テスト環境: Ollama Cloud 経由でクラウド版モデルを使用
    # 事前に `ollama signin` でブラウザ認証が必要
    MODEL_NAME: str = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b-cloud")
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "https://ollama.com")
else:
    # 本番環境: ローカル Ollama（閉域ネットワーク）
    MODEL_NAME: str = os.environ.get("OLLAMA_MODEL", "gpt-oss:120b")
    OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

# --- チャンク分割設定 ---
CHUNK_SIZE: int = int(os.environ.get("CHUNK_SIZE", "130000"))
CHUNK_OVERLAP: int = int(os.environ.get("CHUNK_OVERLAP", "1000"))

# --- 病院設定 ---
HOSPITAL: str = os.environ.get("HOSPITAL", "hanwa")

# --- Agentic Search 設定 ---
MAX_SEARCH_ITERATIONS: int = int(os.environ.get("MAX_SEARCH_ITERATIONS", "3"))

# --- Reflection 設定 ---
MAX_REFLECTION_ITERATIONS: int = int(
    os.environ.get("MAX_REFLECTION_ITERATIONS", "2")
)

# --- LLM 共通オプション ---
LLM_OPTIONS: dict = {
    "temperature": 0.1,
    "top_p": 0.92,
    "repeat_penalty": 1.2,
    "num_ctx": 8192,
    "num_predict": 4096,
}
