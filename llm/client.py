"""Ollama クライアントモジュール。

Ollama Python SDK を直接使用して LLM を呼び出す。
LangChain は使用せず、ollama パッケージのみに依存する。

環境切替:
    production: ローカル Ollama（http://localhost:11434）
    test: Ollama Cloud（https://ollama.com、API キー認証）
"""

import json
import re

import httpx
from ollama import ChatResponse, Client

from config.settings import (
    ENV,
    JSON_TEMPERATURE,
    LLM_OPTIONS,
    MODEL_NAME,
    OLLAMA_BASE_URL,
)


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
                read=600.0,  # ローカル大規模モデルの生成待ち: 10分
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
    think: bool = False,
    options_override: dict | None = None,
) -> str:
    """Ollama にチャットリクエストを送信してテキスト応答を返す。

    Args:
        prompt: ユーザープロンプト。
        system: システムプロンプト（オプション）。
        format_schema: 構造化出力用の JSON Schema（オプション）。
        temperature: 温度パラメータの上書き（オプション）。
        think: 思考モード。gpt-oss は format と排他、Qwen3 系は既定で
            thinking が有効なため、いずれのモデルでも構造化出力時は False を
            指定して reasoning trace の混入を防ぐ（既定 False）。
        options_override: num_ctx 等をノード単位で上書きする辞書（オプション）。

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
    if options_override:
        options.update(options_override)

    kwargs: dict = {
        "model": MODEL_NAME,
        "messages": messages,
        "options": options,
        "keep_alive": "60m",
        "think": think,
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
) -> ChatResponse:
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


def extract_json(text: str) -> dict | None:
    """LLM 応答テキストから JSON オブジェクトを堅牢に抽出する。

    gpt-oss は format 指定時でも reasoning trace やコードフェンス、
    説明文を伴って JSON を返すことがあるため、複数の戦略を順に試す:
    (1) ```json ... ``` コードフェンス内、(2) 最初の平衡した {...}、
    (3) テキスト全体。最初に json.loads できた dict を返す。

    Args:
        text: LLM の応答テキスト。

    Returns:
        パースできた dict。いずれも失敗した場合は None。
    """
    if not text:
        return None

    candidates: list[str] = []

    # 1. コードフェンス内の JSON
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        candidates.append(fence.group(1))

    # 2. 最初の平衡した {...}
    balanced = _first_balanced_object(text)
    if balanced:
        candidates.append(balanced)

    # 3. テキスト全体
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(obj, dict):
            return obj

    return None


def chat_json(
    prompt: str,
    schema: dict,
    *,
    max_retries: int = 3,
    options_override: dict | None = None,
) -> dict | None:
    """構造化出力を堅牢に取得する（format + extract_json + bounded retry）。

    LLM は format 指定でも JSON 以外を混ぜることがあるため、extract_json で
    抽出し、失敗時はエラー文を最新500字だけ付加して再試行する。
    Ollama Cloud では format が強制されないため retry が特に重要。

    温度はモデル別プロファイルの `JSON_TEMPERATURE` を使う。gpt-oss は 0
    （決定的）だが、Qwen 系は greedy decoding が公式に非推奨のため 0 より
    大きい値になる（`config.model_profiles` 参照）。

    Args:
        prompt: ユーザープロンプト。
        schema: 期待する JSON スキーマ（format に渡す）。
        max_retries: 最大試行回数。
        options_override: num_ctx 等の上書き。

    Returns:
        パースできた dict。全試行失敗時は None。
    """
    current = prompt
    for _ in range(max_retries):
        raw = chat(
            current,
            format_schema=schema,
            temperature=JSON_TEMPERATURE,
            think=False,
            options_override=options_override,
        )
        parsed = extract_json(raw)
        if parsed is not None:
            return parsed
        # 再試行: JSON 抽出に失敗した旨を末尾に付加（最新のみ・500字上限）
        snippet = (raw or "")[-500:]
        current = (
            f"{prompt}\n\n# 前回の出力は JSON として解釈できなかった。"
            f"JSON のみを出力すること。前回出力(末尾):\n{snippet}"
        )
    return None


def _first_balanced_object(text: str) -> str | None:
    """テキスト中で最初に出現する平衡した {...} 部分文字列を返す。

    文字列リテラル内の波括弧・エスケープを考慮し、深さ0で閉じた位置までを
    切り出す。見つからない場合は None。
    """
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        else:
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None
