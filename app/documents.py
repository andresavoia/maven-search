"""Page-aware paper chunks and one chunk per deck slide. Page numbers never cross a boundary."""

import re
from pathlib import Path

from pypdf import PdfReader


def read_pdf_pages(path: Path) -> list[dict]:
    reader = PdfReader(str(path))
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            pages.append({"page": index, "text": text})
    return pages


def chunk_page(text: str, page: int, words_per_chunk: int = 120, overlap_words: int = 20) -> list[dict]:
    words = text.split()
    if not words:
        return []
    if len(words) <= words_per_chunk:
        return [{"page": page, "text": text}]
    step = max(1, words_per_chunk - overlap_words)
    chunks = []
    start = 0
    while start < len(words):
        window = words[start : start + words_per_chunk]
        chunks.append({"page": page, "text": " ".join(window)})
        if start + words_per_chunk >= len(words):
            break
        start += step
    return chunks


def passages_from_pdf(path: Path, source_id: str, kind: str, title: str) -> list[dict]:
    passages = []
    for page in read_pdf_pages(path):
        pieces = [page] if kind == "deck" else chunk_page(page["text"], page["page"])
        for piece in pieces:
            number = piece["page"]
            locator = {"slide": number} if kind == "deck" else {"page": number}
            label = f"Slide {number}" if kind == "deck" else f"Page {number}"
            passages.append(
                {
                    "id": f"{source_id}-{kind}-{number}-{len(passages)}",
                    "kind": kind,
                    "source_id": source_id,
                    "source_title": title,
                    "title": f"{title} — {label}",
                    "text": piece["text"],
                    "page": number if kind == "paper" else None,
                    "slide": number if kind == "deck" else None,
                    "locator": locator,
                    "sentences": [{"text": piece["text"], "start": 0, "end": 0}],
                    "start": 0,
                    "end": 0,
                }
            )
    return passages
