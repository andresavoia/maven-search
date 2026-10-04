import json

import numpy as np

from app.chunking import fixed_chunks, prepare_cues
from app.config import INDEX_DIR, TRANSCRIPT_PATH, VIDEO_AUTHOR, VIDEO_ID, VIDEO_TITLE, VIDEO_URL, VIDEO_WHY
from app.corpus import ensure_corpus
from app.documents import passages_from_pdf
from app.embeddings import MiniLMEmbedder
from app.moments import make_beats, segment_moments
from app.transcript import load_or_fetch


def _public_record(record: dict) -> dict:
    kept = {
        "id": record["id"],
        "start": record["start"],
        "end": record["end"],
        "text": record["text"],
        "sentences": record["sentences"],
    }
    if "title" in record:
        kept["title"] = record["title"]
    return kept


def build_index(embedder=None) -> dict:
    transcript = load_or_fetch()
    cues = prepare_cues(transcript["cues"])
    chunks = fixed_chunks(cues)
    beats = make_beats(cues)
    embedder = embedder or MiniLMEmbedder()

    chunk_vectors = embedder.embed([chunk["text"] for chunk in chunks])
    beat_vectors = embedder.embed([beat["text"] for beat in beats]) if beats else np.zeros((0, 1), dtype=np.float32)
    moments = segment_moments(beats, beat_vectors)
    moment_cards = [f"{moment['title']}. {moment['text'][:700]}" for moment in moments]
    moment_vectors = embedder.embed(moment_cards) if moments else np.zeros((0, chunk_vectors.shape[1]), dtype=np.float32)

    for moment in moments:
        moment["beat_vectors_slice"] = [moment["beat_start"], moment["beat_end"]]

    documents: list[dict] = []
    for source in ensure_corpus():
        documents.extend(
            passages_from_pdf(source["path"], source["id"], source["kind"], source["title"])
        )
    doc_vectors = embedder.embed([item["text"] for item in documents]) if documents else np.zeros((0, chunk_vectors.shape[1]), dtype=np.float32)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    meta = {
        "video": {
            "id": VIDEO_ID,
            "url": VIDEO_URL,
            "title": transcript.get("title") or VIDEO_TITLE,
            "author": transcript.get("author") or VIDEO_AUTHOR,
            "why": VIDEO_WHY,
            "cue_count": len(cues),
            "duration": cues[-1]["start"] + cues[-1]["duration"] if cues else 0,
        },
        "chunks": [_public_record(chunk) for chunk in chunks],
        "moments": [
            {
                **_public_record(moment),
                "beat_start": moment["beat_start"],
                "beat_end": moment["beat_end"],
            }
            for moment in moments
        ],
        "documents": documents,
    }
    (INDEX_DIR / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.save(INDEX_DIR / "chunks.npy", chunk_vectors)
    np.save(INDEX_DIR / "moments.npy", moment_vectors)
    np.save(INDEX_DIR / "beats.npy", beat_vectors)
    np.save(INDEX_DIR / "docs.npy", doc_vectors)
    return {
        "chunks": len(chunks),
        "moments": len(moments),
        "beats": len(beats),
        "documents": len(documents),
        "cues": len(cues),
        "transcript": str(TRANSCRIPT_PATH),
    }


if __name__ == "__main__":
    stats = build_index()
    print(json.dumps(stats, indent=2))
