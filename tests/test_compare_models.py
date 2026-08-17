"""モデル比較ハーネスのテスト（LLM 呼び出しを伴わない部分）。"""

from pathlib import Path

import pytest

from scripts.compare_models import build_report, parse_model_spec


class TestParseModelSpec:
    """``モデル名[@接続先URL]`` の解析。"""

    def test_splits_model_and_base_url(self):
        spec = parse_model_spec("qwen3.8:27b@http://localhost:11434")
        assert spec.model == "qwen3.8:27b"
        assert spec.base_url == "http://localhost:11434"

    def test_defaults_to_local_ollama_when_url_omitted(self):
        spec = parse_model_spec("qwen3.8:27b")
        assert spec.base_url == "http://localhost:11434"
        assert spec.env == "production"

    def test_cloud_host_resolves_to_test_env(self):
        """Ollama Cloud 宛ては ENV=test として解決される（認証経路が異なるため）。"""
        spec = parse_model_spec("gpt-oss:120b-cloud@https://ollama.com")
        assert spec.env == "test"

    def test_slug_is_filesystem_safe(self):
        """モデル名の ':' 等がファイル名安全な形に変換される。"""
        assert parse_model_spec("qwen3.8:27b").slug == "qwen3.8_27b"

    def test_empty_model_name_is_rejected(self):
        with pytest.raises(ValueError):
            parse_model_spec("@http://localhost:11434")


class TestBuildReport:
    """比較レポートの生成。"""

    @staticmethod
    def _result(model: str, summary: str, ok: bool = True) -> dict:
        return {
            "model": model,
            "base_url": "http://localhost:11434",
            "ok": ok,
            "error": None if ok else "boom",
            "final_summary": summary,
            "review_flags": ["sec1: 未支持"],
            "elapsed_sec": 12.3,
            "llm_options": {"temperature": 0.7},
            "json_temperature": 0.1,
            "section_results": {
                "sec1": {"body": summary, "missing": [], "review_flag": False}
            },
        }

    def test_report_contains_all_models_and_bodies(self):
        report = build_report(
            [
                self._result("gpt-oss:120b", "サマリーA"),
                self._result("qwen3.8:27b", "サマリーB"),
            ],
            input_path=Path("input.md"),
        )
        assert "gpt-oss:120b" in report
        assert "qwen3.8:27b" in report
        assert "サマリーA" in report
        assert "サマリーB" in report
        # 比較条件が後から追えるよう実行パラメータを載せる。
        assert "llm_options" in report

    def test_failed_model_is_reported_without_breaking_report(self):
        """1 モデルが失敗してもレポート生成は完了する。"""
        report = build_report(
            [
                self._result("gpt-oss:120b", "サマリーA"),
                self._result("qwen3.8:27b", "", ok=False),
            ],
            input_path=Path("input.md"),
        )
        assert "NG" in report
        assert "boom" in report
