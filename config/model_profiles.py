"""モデル別 LLM チューニングプロファイル。

Ollama で利用する LLM は、モデルごとに推奨サンプリングパラメータと
greedy decoding の可否が異なる。本モジュールはモデル名からプロファイルを
解決し、`config.settings` がそれを `LLM_OPTIONS` として公開する。
これにより `.env` の `OLLAMA_MODEL` を差し替えるだけで、そのモデルに
適したパラメータが自動で適用される（呼び出し側のコード変更は不要）。

アーキテクチャ上の位置:
    config.settings → 本モジュール → llm.client（options として消費）
    本モジュールは settings を import しない（循環参照の回避）。

主要クラス/関数:
    ModelProfile: モデル固有のチューニング設定
    resolve_profile: モデル名からプロファイルを解決する
"""

from dataclasses import dataclass

# 全モデル共通の生成設定。
# コンテキスト長・生成長はモデル比較の公平性を保つため意図的に共通化し、
# モデル固有として扱うのはサンプリング系パラメータのみとする。
_COMMON_OPTIONS: dict = {
    "num_ctx": 8192,
    "num_predict": 4096,
}


@dataclass(frozen=True)
class ModelProfile:
    """モデル固有の LLM チューニング設定。

    Attributes:
        sampling: モデル推奨のサンプリングパラメータ。`_COMMON_OPTIONS` と
            マージして Ollama の options に渡される。
        json_temperature: 構造化出力（`chat_json`）で使用する温度。
            greedy decoding（温度0）で無限反復・品質劣化を起こすモデルでは
            0 より大きい値を指定する。
    """

    sampling: dict
    json_temperature: float

    @property
    def options(self) -> dict:
        """Ollama の options に渡す完成形の設定辞書。

        共通設定にモデル固有のサンプリング設定を重ねた新しい辞書を返す。
        呼び出し側が破壊的に更新してもプロファイルは汚染されない。
        """
        return {**_COMMON_OPTIONS, **self.sampling}


# gpt-oss 系: 本システムで実績のある設定。低温＋反復ペナルティで
# 医療記録からの抽出的要約の事実性と安定性を担保する。
_GPT_OSS_PROFILE = ModelProfile(
    sampling={
        "temperature": 0.1,
        "top_p": 0.92,
        "repeat_penalty": 1.2,
    },
    # gpt-oss は温度0でも反復に陥らないため、構造化出力は決定的に取る。
    json_temperature=0.0,
)

# Qwen3 系（qwen3.5 / qwen3.6 / qwen3.8）: 公式の非思考モード推奨値。
# 公式は「repetition_penalty を 1.0 から上げると品質が劣化する」「greedy
# decoding は無限反復を招く」と明記しているため、反復抑制は presence_penalty
# に委ね、repeat_penalty は 1.0 に据え置く。
# 出典: https://huggingface.co/Qwen/Qwen3.8-27B
_QWEN3_PROFILE = ModelProfile(
    sampling={
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "repeat_penalty": 1.0,
        "presence_penalty": 1.5,
    },
    # 温度0は公式が非推奨のため、構造化出力でも greedy を避けつつ
    # 抽出タスクとして再現性を保てる下限値を使う。
    json_temperature=0.1,
)

# モデルファミリ名 → プロファイル。
# キーはモデル名のファミリ部分（"gpt-oss:120b-cloud" → "gpt-oss"）。
MODEL_PROFILES: dict[str, ModelProfile] = {
    "gpt-oss": _GPT_OSS_PROFILE,
    "qwen3.8": _QWEN3_PROFILE,
    "qwen3.6": _QWEN3_PROFILE,
    "qwen3.5": _QWEN3_PROFILE,
    "qwen3": _QWEN3_PROFILE,
}

# 未知モデル用のフォールバック。事実性重視の保守的な設定とし、
# ベンダー固有の癖に依存しないパラメータのみを使う。
DEFAULT_PROFILE = ModelProfile(
    sampling={
        "temperature": 0.2,
        "top_p": 0.9,
    },
    # 未知モデルの greedy 耐性は不明なため、安全側で 0 を避ける。
    json_temperature=0.1,
)


def resolve_profile(model_name: str) -> ModelProfile:
    """モデル名からチューニングプロファイルを解決する。

    量子化・クラウド等のサフィックス（``:120b-cloud`` / ``:27b-q8_0``）を
    無視してファミリ単位で解決するため、タグ違いでも同じプロファイルが
    適用される。未登録のファミリは `DEFAULT_PROFILE` にフォールバックする。

    Args:
        model_name: Ollama のモデル名（例: "qwen3.8:27b"）。

    Returns:
        該当するプロファイル。未登録なら `DEFAULT_PROFILE`。
    """
    family = model_name.split(":")[0].strip().lower()
    return MODEL_PROFILES.get(family, DEFAULT_PROFILE)
