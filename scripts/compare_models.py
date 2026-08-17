"""複数モデルで同一入力から看護サマリーを生成し、結果を比較するハーネス。

同一の医療記録・同一テンプレートに対してモデルだけを差し替えて
`graph.builder` のパイプラインを実行し、生成本文・所要時間・レビューフラグ・
セクション別結果を並べて比較できる形で保存する。

設計上の注意:
    `config.settings.MODEL_NAME` は import 時に確定するモジュール定数のため、
    同一プロセス内でのモデル切替は行わない。モデル 1 つにつき 1 サブプロセス
    （``--worker`` モード）を起動し、環境変数でモデルと接続先を渡す。
    `python-dotenv` は既存の環境変数を上書きしないため、親から渡した設定が
    `.env` より優先される。

実行例:
    uv run python -m scripts.compare_models \\
        --model gpt-oss:120b-cloud@https://ollama.com \\
        --model qwen3.8:27b@http://localhost:11434

主要クラス/関数:
    ModelSpec: 比較対象 1 モデルの指定（モデル名・接続先・ENV）
    run_model: 1 モデル分をサブプロセスで実行する
    build_report: 比較レポート Markdown を生成する
    main: CLI エントリポイント
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# リポジトリルート（本ファイルの親の親）。サブプロセスの cwd に使う。
REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ModelSpec:
    """比較対象 1 モデルの実行指定。

    Attributes:
        model: Ollama のモデル名（例: "qwen3.8:27b"）。
        base_url: Ollama の接続先 URL。
        env: `config.settings` の ENV 値。接続先が Ollama Cloud なら "test"、
            それ以外は "production"。環境別変数の名前解決に使う。
    """

    model: str
    base_url: str
    env: str

    @property
    def slug(self) -> str:
        """ファイル名に使える安全な識別子。"""
        return re.sub(r"[^A-Za-z0-9._-]", "_", self.model)


def parse_model_spec(
    raw: str, default_base_url: str | None = None
) -> ModelSpec:
    """``モデル名[@接続先URL]`` 形式の指定を ModelSpec に変換する。

    接続先が省略された場合は `default_base_url`、それも無ければ Ollama の
    ローカル既定値を使う。接続先ホストが ollama.com なら ENV=test（Cloud）、
    それ以外は ENV=production として解決する。

    Args:
        raw: CLI から渡された指定文字列。
        default_base_url: ``@`` 省略時に使う接続先。

    Returns:
        解析された ModelSpec。

    Raises:
        ValueError: モデル名が空の場合。
    """
    if "@" in raw:
        model, base_url = raw.split("@", 1)
    else:
        model, base_url = raw, (default_base_url or "http://localhost:11434")

    model = model.strip()
    base_url = base_url.strip()
    if not model:
        raise ValueError(f"モデル名が空です: {raw!r}")

    env = "test" if "ollama.com" in base_url else "production"
    return ModelSpec(model=model, base_url=base_url, env=env)


def _build_initial_state(
    context: str, patient_id: str, hospital: str, template_id: str
) -> dict:
    """グラフ実行用の初期 GlobalState を構築する。

    `app.py` の実行経路と同じ初期状態を再現する（比較の前提を揃えるため、
    入力以外のフィールドは全て空で開始する）。

    Args:
        context: 日付チャンク Markdown 形式の医療記録テキスト。
        patient_id: 患者ID。
        hospital: 病院識別子。
        template_id: 使用テンプレートID。

    Returns:
        `graph.state.GlobalState` に対応する初期状態辞書。
    """
    return {
        "patient_id": patient_id,
        "raw_context": context,
        "hospital": hospital,
        "template_id": template_id,
        "template": {},
        "routing": {},
        "summary_header": "",
        "chunks": [],
        "grep_index": [],
        "total_tokens": 0,
        "section_results": {},
        "draft_summary": "",
        "final_summary": "",
        "review_flags": [],
        "error": None,
    }


def _run_worker(args: argparse.Namespace) -> int:
    """ワーカーモード: 現在の環境変数の設定でグラフを 1 回実行する。

    親プロセスが環境変数でモデル・接続先を指定済みである前提で動作し、
    結果を JSON ファイルに書き出す。例外はメッセージを結果に含めて返し、
    1 モデルの失敗が比較全体を止めないようにする。

    Args:
        args: `--out` / `--input` / `--template` / `--patient-id` を含む引数。

    Returns:
        プロセス終了コード（成功 0 / 失敗 1）。
    """
    # settings は環境変数確定後に import する必要があるためここで読み込む。
    from config.settings import (
        HOSPITAL,
        JSON_TEMPERATURE,
        LLM_OPTIONS,
        MODEL_NAME,
        OLLAMA_BASE_URL,
    )
    from graph.builder import build_nursing_summary_graph

    context = Path(args.input).read_text(encoding="utf-8")
    template_id = args.template or HOSPITAL

    payload: dict = {
        "model": MODEL_NAME,
        "base_url": OLLAMA_BASE_URL,
        "llm_options": LLM_OPTIONS,
        "json_temperature": JSON_TEMPERATURE,
        "template_id": template_id,
        "patient_id": args.patient_id,
    }

    started = time.monotonic()
    try:
        graph = build_nursing_summary_graph()
        result = graph.invoke(
            _build_initial_state(
                context, args.patient_id, HOSPITAL, template_id
            )
        )
        payload.update(
            {
                "ok": result.get("error") is None,
                "error": result.get("error"),
                "final_summary": result.get("final_summary", ""),
                "review_flags": result.get("review_flags", []),
                "total_tokens": result.get("total_tokens", 0),
                "section_results": {
                    key: {
                        "body": value.get("body", ""),
                        "missing": value.get("missing", []),
                        "review_flag": value.get("review_flag", False),
                        "cited_dates": value.get("cited_dates", []),
                        "search_steps": len(value.get("search_trace", [])),
                    }
                    for key, value in (
                        result.get("section_results") or {}
                    ).items()
                },
            }
        )
    except Exception as e:  # noqa: BLE001 - 失敗も比較結果として記録する
        payload.update(
            {
                "ok": False,
                "error": f"{type(e).__name__}: {e}",
                "final_summary": "",
                "review_flags": [],
                "section_results": {},
            }
        )

    payload["elapsed_sec"] = round(time.monotonic() - started, 1)
    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0 if payload.get("ok") else 1


def run_model(
    spec: ModelSpec,
    *,
    input_path: Path,
    out_dir: Path,
    template_id: str | None,
    patient_id: str,
    timeout: int,
) -> dict:
    """1 モデル分の生成をサブプロセスで実行し結果 dict を返す。

    環境別変数（``OLLAMA_MODEL_<ENV>``）を明示的に設定することで、`.env` に
    既存の同名設定があっても確実に上書きする。

    Args:
        spec: 実行するモデルの指定。
        input_path: 医療記録テキストのパス。
        out_dir: 結果 JSON の出力先ディレクトリ。
        template_id: テンプレートID（None なら HOSPITAL 環境変数）。
        patient_id: 患者ID。
        timeout: サブプロセスのタイムアウト秒数。

    Returns:
        ワーカーが書き出した結果 dict。タイムアウト・異常終了時は
        ``ok=False`` と error を含む最小限の dict。
    """
    out_path = out_dir / f"{spec.slug}.json"

    env = os.environ.copy()
    env["ENV"] = spec.env
    # 環境別変数が最優先で解決されるため、こちらを直接指定する。
    env[f"OLLAMA_MODEL_{spec.env.upper()}"] = spec.model
    env[f"OLLAMA_BASE_URL_{spec.env.upper()}"] = spec.base_url
    # 共通変数も揃えておき、解決順序が変わっても同じモデルを指すようにする。
    env["OLLAMA_MODEL"] = spec.model
    env["OLLAMA_BASE_URL"] = spec.base_url

    cmd = [
        sys.executable,
        "-m",
        "scripts.compare_models",
        "--worker",
        "--input",
        str(input_path),
        "--out",
        str(out_path),
        "--patient-id",
        patient_id,
    ]
    if template_id:
        cmd += ["--template", template_id]

    print(f"[run] {spec.model} @ {spec.base_url} (ENV={spec.env}) ...")
    started = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=env,
            timeout=timeout,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        elapsed = round(time.monotonic() - started, 1)
        print(f"[timeout] {spec.model}: {timeout}s 超過")
        return {
            "model": spec.model,
            "base_url": spec.base_url,
            "ok": False,
            "error": f"timeout ({timeout}s)",
            "final_summary": "",
            "review_flags": [],
            "section_results": {},
            "elapsed_sec": elapsed,
        }

    if out_path.exists():
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    else:
        # ワーカーが結果を書く前に落ちたケース（import エラー等）。
        payload = {
            "model": spec.model,
            "base_url": spec.base_url,
            "ok": False,
            "error": (proc.stderr or proc.stdout or "unknown error")[-2000:],
            "final_summary": "",
            "review_flags": [],
            "section_results": {},
            "elapsed_sec": round(time.monotonic() - started, 1),
        }
        out_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    status = "ok" if payload.get("ok") else f"NG ({payload.get('error')})"
    print(f"[done] {spec.model}: {payload.get('elapsed_sec')}s / {status}")
    return payload


def build_report(results: list[dict], *, input_path: Path) -> str:
    """比較レポート Markdown を生成する。

    サマリ表（所要時間・文字数・レビューフラグ数）に続けて、モデルごとの
    生成本文全文とセクション別の充足状況を並べる。

    Args:
        results: `run_model` の戻り値のリスト（比較対象の順）。
        input_path: 入力に使った医療記録のパス。

    Returns:
        Markdown 形式のレポート文字列。
    """
    lines: list[str] = []
    lines.append("# モデル比較レポート（看護サマリー生成）")
    lines.append("")
    lines.append(f"- 入力: `{input_path}`")
    lines.append(f"- 生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # --- サマリ表 ---
    lines.append("## 1. サマリ")
    lines.append("")
    lines.append(
        "| モデル | 接続先 | 結果 | 所要時間 | 本文文字数 | "
        "セクション数 | レビューフラグ |"
    )
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        summary = r.get("final_summary") or ""
        status = "OK" if r.get("ok") else "NG"
        lines.append(
            f"| `{r.get('model')}` | {r.get('base_url')} | {status} | "
            f"{r.get('elapsed_sec')}s | {len(summary)} | "
            f"{len(r.get('section_results') or {})} | "
            f"{len(r.get('review_flags') or [])} |"
        )
    lines.append("")

    # --- 実行パラメータ（比較条件の明示） ---
    lines.append("## 2. 実行パラメータ")
    lines.append("")
    for r in results:
        lines.append(f"### `{r.get('model')}`")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(
                {
                    "llm_options": r.get("llm_options"),
                    "json_temperature": r.get("json_temperature"),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        lines.append("```")
        lines.append("")

    # --- セクション別の充足状況 ---
    lines.append("## 3. セクション別の充足状況")
    lines.append("")
    section_keys: list[str] = []
    for r in results:
        for key in r.get("section_results") or {}:
            if key not in section_keys:
                section_keys.append(key)
    if section_keys:
        header = "| セクション | " + " | ".join(
            f"`{r.get('model')}` 文字数 / missing" for r in results
        )
        lines.append(header + " |")
        lines.append("|---" * (len(results) + 1) + "|")
        for key in section_keys:
            cells = []
            for r in results:
                sec = (r.get("section_results") or {}).get(key)
                if sec is None:
                    cells.append("—")
                else:
                    cells.append(
                        f"{len(sec.get('body', ''))} / "
                        f"{len(sec.get('missing') or [])}"
                    )
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
        lines.append("")

    # --- 生成本文全文 ---
    lines.append("## 4. 生成された看護サマリー（全文）")
    lines.append("")
    for r in results:
        lines.append(f"### `{r.get('model')}`")
        lines.append("")
        if r.get("error"):
            lines.append(f"> エラー: {r.get('error')}")
            lines.append("")
        body = r.get("final_summary") or "(生成なし)"
        lines.append("```text")
        lines.append(body)
        lines.append("```")
        lines.append("")
        flags = r.get("review_flags") or []
        if flags:
            lines.append(f"レビューフラグ ({len(flags)}件):")
            lines.append("")
            for flag in flags:
                lines.append(f"- {flag}")
            lines.append("")

    return "\n".join(lines)


def _resolve_input(input_arg: str | None, out_dir: Path) -> Path:
    """入力ファイルを解決する（未指定ならテスト用サンプルを書き出す）。

    比較の再現性のため、既定入力を使った場合もその内容を出力先に保存する。

    Args:
        input_arg: `--input` の値（None 可）。
        out_dir: 出力先ディレクトリ。

    Returns:
        医療記録テキストのパス。
    """
    if input_arg:
        return Path(input_arg).resolve()

    from tests.conftest import SAMPLE_MEDICAL_RECORD

    path = out_dir / "input.md"
    path.write_text(SAMPLE_MEDICAL_RECORD, encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    """CLI エントリポイント（オーケストレータ / ワーカーを分岐）。

    Args:
        argv: コマンドライン引数（None なら sys.argv）。

    Returns:
        プロセス終了コード。全モデル成功で 0、1 つでも失敗なら 1。
    """
    parser = argparse.ArgumentParser(
        description="複数モデルで看護サマリーを生成し比較する"
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        metavar="MODEL[@BASE_URL]",
        help="比較対象モデル（複数指定可）。例: qwen3.8:27b@http://localhost:11434",
    )
    parser.add_argument(
        "--input", help="医療記録テキストのパス（省略時はサンプル）"
    )
    parser.add_argument(
        "--template", help="テンプレートID（省略時は HOSPITAL）"
    )
    parser.add_argument("--patient-id", default="TEST001", help="患者ID")
    parser.add_argument(
        "--output-dir",
        default="output/model-comparison",
        help="結果の出力先ディレクトリ",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="1 モデルあたりのタイムアウト秒数（既定 3600）",
    )
    # 内部用（サブプロセス実行）
    parser.add_argument(
        "--worker", action="store_true", help=argparse.SUPPRESS
    )
    parser.add_argument("--out", help=argparse.SUPPRESS)

    args = parser.parse_args(argv)

    if args.worker:
        return _run_worker(args)

    if not args.model:
        parser.error("--model を 1 つ以上指定してください")

    run_dir = Path(args.output_dir) / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    input_path = _resolve_input(args.input, run_dir)
    specs = [parse_model_spec(raw) for raw in args.model]

    results = [
        run_model(
            spec,
            input_path=input_path,
            out_dir=run_dir,
            template_id=args.template,
            patient_id=args.patient_id,
            timeout=args.timeout,
        )
        for spec in specs
    ]

    report_path = run_dir / "report.md"
    report_path.write_text(
        build_report(results, input_path=input_path), encoding="utf-8"
    )
    print(f"\nレポート: {report_path}")

    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
