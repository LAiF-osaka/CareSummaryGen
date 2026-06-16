"""看護サマリー生成 API サーバー。

LangGraph + Ollama SDK + FastAPI による Agentic Search パイプライン。
医療記録テキストを受け取り、テンプレートに沿って看護サマリーを自動生成する。

起動:
    uv run uvicorn app:app --reload --port 5000
"""

import re
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from adapters.pipeline import build_context_from_db
from config.settings import HOSPITAL, MODEL_NAME
from graph.builder import build_nursing_summary_graph
from templates_loader.loader import list_templates, load_template

# --- リクエスト/レスポンスモデル ---


class AskRequest(BaseModel):
    """看護サマリー生成リクエスト（テキスト入力）。"""

    context: str = Field(..., min_length=1, description="医療記録テキスト")
    patient_id: str = Field(default="unknown", description="患者ID")
    template_id: str | None = Field(
        default=None,
        description="テンプレートID（省略時は HOSPITAL 環境変数）",
    )


class AskResponse(BaseModel):
    """看護サマリー生成レスポンス。"""

    answer: str = Field(description="生成された看護サマリー")
    template_id: str = Field(description="使用されたテンプレートID")
    review_flags: list[str] = Field(
        default_factory=list,
        description="人手レビューが必要なセクションキー・未支持claim",
    )


class IngestRequest(BaseModel):
    """DB取得ベースの看護サマリー生成リクエスト。"""

    patient_id: str = Field(..., description="患者ID")
    encounter_id: str = Field(..., description="入院ID")
    query_spec_id: str = Field(
        default="sql_sample", description="取得仕様ID（query_specs/<id>.yaml）"
    )
    template_id: str | None = Field(
        default=None, description="テンプレートID（省略時は HOSPITAL）"
    )


class TemplateInfo(BaseModel):
    """テンプレート情報。"""

    id: str
    name: str
    description: str = ""


class HealthResponse(BaseModel):
    """ヘルスチェックレスポンス。"""

    status: str
    model: str


# --- アプリケーション ---


@asynccontextmanager
async def lifespan(app: FastAPI):
    """起動時にグラフを構築し取得アダプタを登録する。"""
    print(f"看護サマリー生成 API を起動中... モデル: {MODEL_NAME}")
    # 取得アダプタ実装を register_adapter で登録する（import で副作用）
    import adapters.sql_source  # noqa: F401

    app.state.graph = build_nursing_summary_graph()
    print("LangGraph グラフの構築完了")
    yield
    print("API サーバーを停止します")


app = FastAPI(
    title="CareSummaryGen API",
    description="医療記録から看護サマリーを自動生成する API",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", response_model=HealthResponse)
async def index() -> HealthResponse:
    """ヘルスチェック。"""
    return HealthResponse(status="ok", model=MODEL_NAME)


@app.get("/templates", response_model=list[TemplateInfo])
async def get_templates() -> list[TemplateInfo]:
    """利用可能なテンプレート一覧を返す。"""
    return [TemplateInfo(**t) for t in list_templates()]


def _run_graph(context: str, patient_id: str, template_id: str) -> AskResponse:
    """context を受けて v2 グラフを実行し AskResponse を返す共通処理。

    initial_state（GlobalState）構築をここに一元化し、/ask と /ingest で共有する。

    Args:
        context: 日付チャンク Markdown 形式の医療記録テキスト。
        patient_id: 患者ID。
        template_id: 使用テンプレートID。

    Returns:
        生成された看護サマリーのレスポンス。
    """
    initial_state = {
        "patient_id": patient_id,
        "raw_context": context,
        "hospital": HOSPITAL,
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

    try:
        result = app.state.graph.invoke(initial_state)
    except Exception as e:
        error_detail = f"エラーが発生しました: {e}"
        print(f"{error_detail}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return AskResponse(
        answer=result["final_summary"],
        template_id=template_id,
        review_flags=result.get("review_flags", []),
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    """テキスト入力から看護サマリーを生成する。

    医療記録テキスト（日付チャンク Markdown）を受け取り、テンプレートに
    沿って agentic search（v2）で看護サマリーを生成する。
    """
    template_id = req.template_id or HOSPITAL
    try:
        load_template(template_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return _run_graph(req.context, req.patient_id, template_id)


@app.post("/ingest", response_model=AskResponse)
async def ingest_endpoint(req: IngestRequest) -> AskResponse:
    """DB取得から看護サマリーを生成する。

    query_spec に基づき DB から取得・正規化・整形して context を構築し、
    /ask と同一の v2 グラフで看護サマリーを生成する。認証は本実装では
    未対応（閉域運用前提・docs/db-input-design.md §7.3）。
    """
    template_id = req.template_id or HOSPITAL
    try:
        load_template(template_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 患者ID・入院IDの最小入力検証（英数字・ハイフン・アンダースコア）
    for value, name in (
        (req.patient_id, "patient_id"),
        (req.encounter_id, "encounter_id"),
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise HTTPException(
                status_code=400, detail=f"{name} の形式が不正です"
            )

    try:
        context = build_context_from_db(
            req.patient_id, req.encounter_id, req.query_spec_id
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return _run_graph(context, req.patient_id, template_id)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
