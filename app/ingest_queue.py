"""Ingest papers and decks off the request path. Status becomes indexed only after the upsert."""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from queue import Queue
from urllib.parse import urlparse

import httpx

from app.config import ROOT
from app.documents import passages_from_pdf

INBOX = ROOT / "data" / "inbox"
STATUS_PATH = ROOT / "data" / "sources.json"

_queue: Queue = Queue()
_worker_started = False
_lock = threading.Lock()


def _read_status() -> list[dict]:
    if not STATUS_PATH.exists():
        return []
    return json.loads(STATUS_PATH.read_text(encoding="utf-8"))


def _write_status(rows: list[dict]) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(rows), encoding="utf-8")


def _update(source_id: str, **fields) -> None:
    with _lock:
        rows = _read_status()
        for row in rows:
            if row["id"] == source_id:
                row.update(fields)
                break
        _write_status(rows)


def list_sources() -> list[dict]:
    with _lock:
        return _read_status()


def enqueue_document(uri: str, kind: str, title: str) -> dict:
    source_id = f"doc_{uuid.uuid4().hex[:8]}"
    row = {"id": source_id, "kind": kind, "title": title, "uri": uri, "status": "pending", "pct": 0}
    with _lock:
        rows = _read_status()
        rows.append(row)
        _write_status(rows)
    _queue.put(row)
    return {"id": source_id, "status": "pending", "kind": kind}


def _download(uri: str, source_id: str) -> Path:
    INBOX.mkdir(parents=True, exist_ok=True)
    destination = INBOX / f"{source_id}.pdf"
    if uri.startswith("file://"):
        source = Path(urlparse(uri).path)
        destination.write_bytes(source.read_bytes())
        return destination
    if uri.startswith("/"):
        destination.write_bytes(Path(uri).read_bytes())
        return destination
    response = httpx.get(uri, follow_redirects=True, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def _run(row: dict) -> None:
    from app.search import get_index

    source_id = row["id"]
    try:
        _update(source_id, status="parsing", pct=20)
        path = _download(row["uri"], source_id)
        _update(source_id, status="chunking", pct=45)
        passages = passages_from_pdf(path, source_id, row["kind"], row["title"])
        if not passages:
            raise ValueError("No text extracted from document")
        _update(source_id, status="embedding", pct=70)
        index = get_index()
        vectors = index.embedder.embed([item["text"] for item in passages])
        index.add_documents(passages, vectors)
        _update(source_id, status="indexed", pct=100)
    except Exception as exc:
        _update(source_id, status="failed", pct=0, error=str(exc))


def _loop() -> None:
    while True:
        row = _queue.get()
        try:
            _run(row)
        finally:
            _queue.task_done()


def start_worker() -> None:
    global _worker_started
    with _lock:
        if _worker_started:
            return
        _worker_started = True
    for row in _read_status():
        if row.get("status") not in {"indexed", "failed"}:
            row["status"] = "pending"
            row["pct"] = 0
            _queue.put(row)
    threading.Thread(target=_loop, name="ingest-worker", daemon=True).start()
