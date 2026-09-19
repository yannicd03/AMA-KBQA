"""FastAPI application for the React demo frontend.

Routes (all under ``/api``): ``health``, ``meta``, ``runs`` (create, cancel,
record, SSE events, trace) and ``sessions/{id}/reset``. The behaviour mirrors the
Streamlit chat page (``ama_kbqa/frontend/chat.py``); see ``runs.py`` for the
run lifecycle and ``stdout_router.py`` for the live-log capture.

Run it through ``ama-kbqa-api`` (uvicorn, exactly one worker: every run,
session and multiturn agent lives in this process's memory).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Optional

from fastapi import APIRouter, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from ama_kbqa.api import meta, stdout_router
from ama_kbqa.api.runs import ApiError, RunManager, sse_stream


class RunRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    agent: str
    model: str
    # Only read when `model` is the picker's free-text entry; see
    # meta.custom_model_option for the shape rules.
    custom_model: Optional[str] = Field(default=None, max_length=200)
    temperature: float = Field(ge=0.0, le=2.0)


def _validation_detail(exc: RequestValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err.get("loc", ()) if p != "body")
        parts.append(f"{loc}: {err.get('msg')}" if loc else str(err.get("msg")))
    return "; ".join(parts) or "Invalid request."


def build_router(manager: RunManager) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    # Plain `def`: FastAPI runs it in its threadpool, which matters because a
    # cache miss fetches the KIT /models list over HTTP.
    @router.get("/meta")
    def get_meta() -> dict:
        return meta.build_meta()

    @router.post("/runs", status_code=201)
    async def create_run(body: RunRequest) -> dict:
        run = await manager.create_run(
            session_id=body.session_id,
            question=body.question,
            agent_name=body.agent,
            model=body.model,
            custom_model=body.custom_model,
            temperature=body.temperature,
        )
        return {"run_id": run.run_id, "continuation": run.continuation}

    # 202, not 204: cancelling only *asks* the run to stop. The agent notices
    # the token at its next checkpoint, so the work is still in flight when
    # this answers — 204 would claim it was already over. Unknown, evicted and
    # already-finished runs answer 202 too: stopping something that is no
    # longer running is the outcome the caller wanted, not an error.
    @router.post("/runs/{run_id}/cancel", status_code=202)
    async def cancel_run(run_id: str) -> dict:
        return manager.cancel_run(run_id)

    @router.post("/sessions/{session_id}/reset", status_code=204)
    async def reset_session(session_id: str) -> Response:
        manager.reset_session(session_id)
        return Response(status_code=204)

    @router.get("/runs/{run_id}/events")
    async def run_events(run_id: str) -> StreamingResponse:
        run = manager.get_run(run_id)
        return StreamingResponse(
            sse_stream(run),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.get("/runs/{run_id}")
    async def get_run(run_id: str) -> dict:
        return manager.get_run(run_id).record()

    @router.get("/runs/{run_id}/trace")
    async def get_trace(run_id: str) -> dict:
        run = manager.get_run(run_id)
        if not run.finished:
            raise ApiError(409, "The trace is available once the run has finished.")
        return run.trace_payload()

    return router


def create_app(manager: Optional[RunManager] = None) -> FastAPI:
    manager = manager or RunManager()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        proxy = stdout_router.install()
        # Pin the default temperature before any run's apply_chat_settings
        # overwrites the env var it is read from.
        meta.default_temperature()
        try:
            yield
        finally:
            await manager.shutdown()
            stdout_router.uninstall(proxy)

    app = FastAPI(
        title="AMA-KBQA API",
        lifespan=lifespan,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        redoc_url=None,
    )
    app.state.manager = manager

    @app.exception_handler(ApiError)
    async def _api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    # The contract answers malformed bodies with 400 {"detail": str}, not
    # FastAPI's default 422 with a list.
    @app.exception_handler(RequestValidationError)
    async def _invalid(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": _validation_detail(exc)})

    app.include_router(build_router(manager), prefix="/api")
    return app


app = create_app()
