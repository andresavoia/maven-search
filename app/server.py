import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import ROOT
from app.ingest_queue import enqueue_document, list_sources, start_worker
from app.search import get_index, video_payload


def load_env() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


@asynccontextmanager
async def lifespan(_app: FastAPI):
    load_env()
    start_worker()
    yield


app = FastAPI(title="Semantic Search vs Moment RAG", lifespan=lifespan)
STATIC = ROOT / "static"


class AskRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)


class DocumentRequest(BaseModel):
    uri: str = Field(min_length=3, max_length=2000)
    kind: str
    title: str = Field(min_length=1, max_length=300)


def require_admin(authorization: str | None) -> None:
    token = os.environ.get("ADMIN_TOKEN", "")
    if not token or authorization != f"Bearer {token}":
        raise HTTPException(status_code=401, detail="Missing or bad admin token")


@app.get("/api/health")
def health() -> dict:
    ready = (ROOT / "data" / "index" / "meta.json").exists()
    return {"ok": True, "indexed": ready}


@app.get("/api/video")
def video() -> dict:
    try:
        return video_payload()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/documents", status_code=202)
def create_document(
    body: DocumentRequest,
    authorization: str | None = Header(default=None),
) -> dict:
    require_admin(authorization)
    if body.kind not in {"paper", "deck"}:
        raise HTTPException(status_code=400, detail="kind must be paper or deck")
    if not body.uri.startswith(("http://", "https://", "file://", "/")):
        raise HTTPException(status_code=400, detail="uri must be an http(s) URL or a file path")
    return enqueue_document(body.uri, body.kind, body.title)


@app.get("/api/sources")
def sources(authorization: str | None = Header(default=None)) -> dict:
    require_admin(authorization)
    return {"sources": list_sources()}


@app.post("/api/ask")
def ask(body: AskRequest) -> dict:
    try:
        return get_index().ask(body.query)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")


def main() -> None:
    import uvicorn

    uvicorn.run("app.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
