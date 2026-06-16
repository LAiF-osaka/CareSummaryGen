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
# .env で ENV を切り替えるだけで Ollama 設定が一括で入れ替わる。
ENV: str = os.environ.get("ENV", "production").lower()

# 環境ごとのデフォルト値（環境別変数が未設定のときに使用）。
#   production: ローカル Ollama（閉域ネットワーク）
#   test: Ollama Cloud（インターネット経由、要 `ollama signin`）
_ENV_DEFAULTS: dict[str, tuple[str, str]] = {
    "production": ("gpt-oss:120b", "http://localhost:11434"),
    "test": ("gpt-oss:120b-cloud", "https://ollama.com"),
}
_default_model, _default_base_url = _ENV_DEFAULTS.get(
    ENV, _ENV_DEFAULTS["production"]
)


def _resolve(name: str, default: str) -> str:
    """環境別変数を解決する。

    優先順位: 環境別変数 ``NAME_<ENV>`` → 共通変数 ``NAME`` → ``default``。
    これにより .env に両環境の設定を併記しておき、ENV の切替だけで
    対応する設定へ一括で切り替えられる。

    Args:
        name: 環境変数のベース名（例: "OLLAMA_MODEL"）。
        default: いずれも未設定の場合に使用するデフォルト値。

    Returns:
        解決された設定値。
    """
    env_specific = os.environ.get(f"{name}_{ENV.upper()}")
    if env_specific:
        return env_specific
    return os.environ.get(name, default)


# --- Ollama 設定（環境依存） ---
MODEL_NAME: str = _resolve("OLLAMA_MODEL", _default_model)
OLLAMA_BASE_URL: str = _resolve("OLLAMA_BASE_URL", _default_base_url)

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
