"""Ollama クライアントモジュール。

Ollama Python SDK を直接使用して LLM を呼び出す。
LangChain は使用せず、ollama パッケージのみに依存する。

環境切替:
    production: ローカル Ollama（http://localhost:11434）
    test: Ollama Cloud（https://ollama.com、API キー認証）
"""

import os

import httpx
from ollama import Client

from config.settings import ENV, LLM_OPTIONS, MODEL_NAME, OLLAMA_BASE_URL


def _build_client() -> Client:
    """環境に応じた Ollama クライアントを構築する。

    テスト環境では Ollama Cloud への接続用にタイムアウトを短縮し、
    環境変数 OLLAMA_API_KEY が設定されていれば自動的に認証ヘッダーを付与する。
    本番環境ではローカル Ollama 向けに長めのタイムアウトを設定する。
    """
    if ENV == "test":
        # テスト環境: Ollama Cloud
        # OLLAMA_API_KEY は Ollama SDK が自動的に認証ヘッダーに付与する
        return Client(
            host=OLLAMA_BASE_URL,
            timeout=httpx.Timeout(
                connect=30.0,
                read=300.0,  # クラウドはローカルより高速
                write=30.0,
                pool=30.0,
            ),
        )
    else:
        # 本番環境: ローカル Ollama
        return Client(
            host=OLLAMA_BASE_URL,
            timeout=httpx.Timeout(
                connect=30.0,
                read=600.0,  # 120B モデルの生成待ち: 10分
                write=30.0,
                pool=30.0,
            ),
        )


# Ollama クライアント（シングルトン）
ollama_client = _build_client()


def chat(
    prompt: str,
    *,
    system: str = "",
    format_schema: dict | None = None,
    temperature: float | None = None,
) -> str:
    """Ollama にチャットリクエストを送信してテキスト応答を返す。

    Args:
        prompt: ユーザープロンプト。
        system: システムプロンプト（オプション）。
        format_schema: 構造化出力用の JSON Schema（オプション）。
        temperature: 温度パラメータの上書き（オプション）。

    Returns:
        LLM の応答テキスト。
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    options = {**LLM_OPTIONS}
    if temperature is not None:
        options["temperature"] = temperature

    kwargs: dict = {
        "model": MODEL_NAME,
        "messages": messages,
        "options": options,
        "keep_alive": "60m",
    }
    if format_schema is not None:
        kwargs["format"] = format_schema

    response = ollama_client.chat(**kwargs)
    return response.message.content


def chat_with_tools(
    prompt: str,
    tools: list,
    *,
    system: str = "",
) -> "ChatResponse":
    """ツール呼び出し付きでチャットリクエストを送信する。

    Args:
        prompt: ユーザープロンプト。
        tools: Ollama に渡すツール関数のリスト。
        system: システムプロンプト（オプション）。

    Returns:
        Ollama の ChatResponse オブジェクト（tool_calls を含む可能性あり）。
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = ollama_client.chat(
        model=MODEL_NAME,
        messages=messages,
        tools=tools,
        options=LLM_OPTIONS,
        keep_alive="60m",
    )
    return response
