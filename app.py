"""看護サマリー生成 API サーバー。

LangGraph + Ollama SDK + FastAPI による Agentic Search パイプライン。
医療記録テキストを受け取り、テンプレートに沿って看護サマリーを自動生成する。

起動:
    uv run uvicorn app:app --reload --port 5000
"""

import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config.settings import (
    HOSPITAL,
    MAX_REFLECTION_ITERATIONS,
    MAX_SEARCH_ITERATIONS,
    MODEL_NAME,
)
from graph.builder import build_nursing_summary_graph
from templates_loader.loader import list_templates, load_template


# --- リクエスト/レスポンスモデル ---


class AskRequest(BaseModel):
    """看護サマリー生成リクエスト。"""

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
    iteration_count: int = Field(description="Reflection の反復回数")


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
    """起動時にグラフを構築する。"""
    print(f"看護サマリー生成 API を起動中... モデル: {MODEL_NAME}")
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


@app.post(
    "/ask",
    response_model=AskResponse,
)
async def ask(req: AskRequest) -> AskResponse:
    """看護サマリーを生成する。

    医療記録テキストを受け取り、指定テンプレートに沿って
    Agentic Search + Reflection で看護サマリーを生成する。
    """
    template_id = req.template_id or HOSPITAL

    try:
        load_template(template_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # LangGraph グラフの初期状態を構築
    initial_state = {
        "patient_id": req.patient_id,
        "raw_context": req.context,
        "hospital": HOSPITAL,
        "chunks": [],
        "chunk_index": [],
        "summary_header": "",
        "template_id": template_id,
        "template": {},
        "search_plan": [],
        "section_results": {},
        "current_section_idx": 0,
        "search_iteration": 0,
        "max_search_iterations": MAX_SEARCH_ITERATIONS,
        "_search_results": [],
        "_section_sufficient": False,
        "draft_summary": "",
        "reflection_feedback": "",
        "reflection_approved": False,
        "iteration_count": 0,
        "max_iterations": MAX_REFLECTION_ITERATIONS,
        "final_summary": "",
        "error": None,
    }

    try:
        result = app.state.graph.invoke(initial_state)

        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["error"])

        return AskResponse(
            answer=result["final_summary"],
            template_id=result["template_id"],
            iteration_count=result["iteration_count"],
        )

    except HTTPException:
        raise
    except Exception as e:
        error_detail = f"エラーが発生しました: {e}"
        print(f"{error_detail}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5000)
