import time

from fastapi.testclient import TestClient

from app.corpus import _write_pdf
from app.documents import chunk_page, passages_from_pdf
from app.search import _diversify
from app.server import app


def test_page_numbers_stay_on_their_page(tmp_path):
    path = tmp_path / "paper.pdf"
    _write_pdf(path, "Notes", ["alpha only", "beta only", "reward model comparison labels"])
    passages = passages_from_pdf(path, "paper-1", "paper", "Notes")
    reward = [item for item in passages if "reward model" in item["text"]]
    assert reward[0]["page"] == 3
    assert reward[0]["locator"] == {"page": 3}
    assert all("beta" not in item["text"] for item in passages if item["page"] == 1)


def test_slide_numbers_follow_pdf_pages(tmp_path):
    path = tmp_path / "deck.pdf"
    _write_pdf(path, "Slides", ["intro", "pretraining", "jailbreak suffix"])
    slides = passages_from_pdf(path, "deck-1", "deck", "Slides")
    assert [slide["slide"] for slide in slides] == [1, 2, 3]
    assert slides[2]["locator"] == {"slide": 3}


def test_chunk_page_does_not_borrow_another_page():
    first = chunk_page("alpha " * 200, page=1, words_per_chunk=50, overlap_words=10)
    second = chunk_page("beta " * 30, page=2, words_per_chunk=50, overlap_words=10)
    assert first and all(chunk["page"] == 1 for chunk in first)
    assert "beta" not in " ".join(chunk["text"] for chunk in first)
    assert second[0]["page"] == 2


def test_diversify_keeps_one_of_each_kind_first():
    hits = [
        {"kind": "video", "text": "a"},
        {"kind": "video", "text": "b"},
        {"kind": "paper", "text": "c"},
        {"kind": "deck", "text": "d"},
    ]
    picked = _diversify(hits, 3)
    assert [hit["kind"] for hit in picked] == ["video", "paper", "deck"]


def test_document_post_returns_pending_without_parsing(monkeypatch, tmp_path):
    monkeypatch.setenv("ADMIN_TOKEN", "test-token")
    monkeypatch.setattr("app.ingest_queue.STATUS_PATH", tmp_path / "sources.json")
    monkeypatch.setattr("app.ingest_queue.passages_from_pdf", lambda *args, **kwargs: [])

    started = time.perf_counter()
    with TestClient(app) as client:
        missing = client.post(
            "/api/documents",
            json={"uri": "https://example.com/paper.pdf", "kind": "paper", "title": "Survey"},
        )
        bad = client.post(
            "/api/documents",
            headers={"Authorization": "Bearer test-token"},
            json={"uri": "https://example.com/paper.pdf", "kind": "image", "title": "Survey"},
        )
        accepted = client.post(
            "/api/documents",
            headers={"Authorization": "Bearer test-token"},
            json={"uri": "https://example.com/paper.pdf", "kind": "paper", "title": "Survey"},
        )
    assert missing.status_code == 401
    assert bad.status_code == 400
    assert accepted.status_code == 202
    assert accepted.json()["status"] == "pending"
    assert time.perf_counter() - started < 1.0
