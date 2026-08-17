"""モデル別プロファイル解決と設定切替のテスト。

`.env` のモデル指定を差し替えるだけでサンプリングパラメータが
そのモデル向けに切り替わることを検証する。
"""

import importlib

import pytest

from config.model_profiles import (
    DEFAULT_PROFILE,
    MODEL_PROFILES,
    resolve_profile,
)


class TestResolveProfile:
    """モデル名からのプロファイル解決。"""

    @pytest.mark.parametrize(
        "model_name",
        ["gpt-oss:120b", "gpt-oss:120b-cloud", "gpt-oss:20b", "GPT-OSS:120b"],
    )
    def test_gpt_oss_family_resolves_to_same_profile(self, model_name):
        """タグ・大文字小文字が違っても gpt-oss は同一プロファイルになる。"""
        profile = resolve_profile(model_name)
        assert profile is MODEL_PROFILES["gpt-oss"]
        assert profile.sampling["temperature"] == 0.1
        assert profile.sampling["repeat_penalty"] == 1.2
        # gpt-oss は温度0でも安定するため構造化出力は決定的に取る。
        assert profile.json_temperature == 0.0

    @pytest.mark.parametrize(
        "model_name",
        ["qwen3.8:27b", "qwen3.8:27b-q8_0", "qwen3.8:latest", "qwen3.5:9b"],
    )
    def test_qwen3_family_uses_official_recommended_sampling(self, model_name):
        """Qwen3 系は公式推奨（非思考モード）のサンプリング値になる。"""
        profile = resolve_profile(model_name)
        assert profile.sampling["temperature"] == 0.7
        assert profile.sampling["top_p"] == 0.8
        assert profile.sampling["top_k"] == 20
        # 公式が repetition_penalty の引き上げを非推奨としているため 1.0。
        assert profile.sampling["repeat_penalty"] == 1.0

    def test_qwen3_avoids_greedy_decoding_for_structured_output(self):
        """Qwen3 系は greedy decoding が非推奨のため温度0を使わない。"""
        assert resolve_profile("qwen3.8:27b").json_temperature > 0

    def test_unknown_model_falls_back_to_default(self):
        """未登録モデルは既定プロファイルにフォールバックする。"""
        assert resolve_profile("llama4:70b") is DEFAULT_PROFILE
        assert resolve_profile("").json_temperature > 0


class TestProfileOptions:
    """options プロパティの構築。"""

    def test_options_merge_common_settings(self):
        """共通設定（num_ctx / num_predict）がマージされる。"""
        options = resolve_profile("qwen3.8:27b").options
        assert options["num_ctx"] == 8192
        assert options["num_predict"] == 4096
        assert options["temperature"] == 0.7

    def test_options_returns_fresh_dict_each_call(self):
        """呼び出し側が破壊的に更新してもプロファイルが汚染されない。"""
        profile = resolve_profile("gpt-oss:120b")
        first = profile.options
        first["temperature"] = 99.0
        assert profile.options["temperature"] == 0.1

    def test_context_length_is_shared_across_models(self):
        """比較の公平性のため、コンテキスト長はモデル間で共通に保つ。"""
        gpt_oss = resolve_profile("gpt-oss:120b").options
        qwen = resolve_profile("qwen3.8:27b").options
        assert gpt_oss["num_ctx"] == qwen["num_ctx"]
        assert gpt_oss["num_predict"] == qwen["num_predict"]


class TestSettingsModelSwitch:
    """環境変数によるモデル切替が settings に反映されること。"""

    @staticmethod
    def _reload_settings(monkeypatch, model: str):
        """指定モデルで config.settings を再読み込みする。

        `.env` の有無に依存しないよう、解決に関わる環境変数を全て明示する。
        """
        monkeypatch.setenv("ENV", "production")
        monkeypatch.setenv("OLLAMA_MODEL", model)
        monkeypatch.setenv("OLLAMA_MODEL_PRODUCTION", model)
        monkeypatch.setenv(
            "OLLAMA_BASE_URL_PRODUCTION", "http://localhost:11434"
        )

        import config.settings

        return importlib.reload(config.settings)

    @pytest.fixture(autouse=True)
    def _restore_settings(self):
        """テスト後に settings を元の環境変数で復元する。"""
        yield
        import config.settings

        importlib.reload(config.settings)

    def test_switching_to_qwen_changes_llm_options(self, monkeypatch):
        """モデル指定を qwen3.8 にすると LLM_OPTIONS が切り替わる。"""
        settings = self._reload_settings(monkeypatch, "qwen3.8:27b")
        assert settings.MODEL_NAME == "qwen3.8:27b"
        assert settings.LLM_OPTIONS["temperature"] == 0.7
        assert settings.LLM_OPTIONS["repeat_penalty"] == 1.0
        assert settings.JSON_TEMPERATURE > 0

    def test_switching_to_gpt_oss_restores_existing_tuning(self, monkeypatch):
        """gpt-oss 指定では従来どおりのチューニング値が使われる。"""
        settings = self._reload_settings(monkeypatch, "gpt-oss:120b")
        assert settings.MODEL_NAME == "gpt-oss:120b"
        assert settings.LLM_OPTIONS["temperature"] == 0.1
        assert settings.LLM_OPTIONS["top_p"] == 0.92
        assert settings.LLM_OPTIONS["repeat_penalty"] == 1.2
        assert settings.LLM_OPTIONS["num_ctx"] == 8192
        assert settings.LLM_OPTIONS["num_predict"] == 4096
        assert settings.JSON_TEMPERATURE == 0.0
