import json
import re
import threading
from functools import lru_cache

import numpy as np

from app.answer import answer
from app.bm25 import BM25, reciprocal_rank_fusion
from app.config import EXPANSIONS, INDEX_DIR, SAMPLE_QUERIES, TOP_K
from app.embeddings import MiniLMEmbedder, l2_normalize


def _iou(left: dict, right: dict) -> float:
    start = max(left["start"], right["start"])
    end = min(left["end"], right["end"])
    intersection = max(0.0, end - start)
    union = max(left["end"], right["end"]) - min(left["start"], right["start"])
    if union <= 0:
        return 0.0
    return intersection / union


def _scale(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(values)) if len(values) else 0.0
    if peak <= 1e-9:
        return np.zeros_like(values, dtype=np.float32)
    return np.asarray(values, dtype=np.float32) / peak


def dedupe_by_time(hits: list[dict], limit: int) -> list[dict]:
    kept: list[dict] = []
    for hit in hits:
        if any(_iou(hit, existing) > 0.55 for existing in kept):
            continue
        kept.append(hit)
        if len(kept) == limit:
            break
    return kept


def expand_query(query: str) -> str:
    """Spell out acronyms the talk says in full. Retrieval uses the longer form."""
    extras: list[str] = []
    lowered = query.lower()
    for key, phrase in EXPANSIONS.items():
        if re.search(rf"\b{re.escape(key)}\b", lowered):
            extras.append(phrase)
    if not extras:
        return query
    return f"{query} {' '.join(extras)}"


class SearchIndex:
    def __init__(self, embedder) -> None:
        meta = json.loads((INDEX_DIR / "meta.json").read_text(encoding="utf-8"))
        self.video = meta["video"]
        self.chunks = meta["chunks"]
        self.moments = meta["moments"]
        self.chunk_vectors = l2_normalize(np.load(INDEX_DIR / "chunks.npy"))
        self.moment_vectors = l2_normalize(np.load(INDEX_DIR / "moments.npy"))
        self.beat_vectors = l2_normalize(np.load(INDEX_DIR / "beats.npy"))
        self.documents = meta.get("documents") or []
        docs_path = INDEX_DIR / "docs.npy"
        if self.documents and docs_path.exists():
            self.doc_vectors = l2_normalize(np.load(docs_path))
        else:
            self.documents = []
            self.doc_vectors = np.zeros((0, self.moment_vectors.shape[1]), dtype=np.float32)
        self.embedder = embedder
        self._lock = threading.Lock()
        self._chunk_bm25 = BM25([chunk["text"] for chunk in self.chunks])
        self._moment_bm25 = BM25([f"{moment['title']} {moment['text']}" for moment in self.moments])
        self._rebuild_cross_bm25()

    def _query_vector(self, query: str) -> np.ndarray:
        return l2_normalize(self.embedder.embed([query]))[0]

    def baseline(self, query: str, limit: int = TOP_K) -> list[dict]:
        query_vector = self._query_vector(query)
        dense = self.chunk_vectors @ query_vector
        order = np.argsort(-dense)
        hits = []
        for index in order:
            chunk = self.chunks[int(index)]
            hits.append({**chunk, "score": float(dense[index]), "kind": "chunk"})
        return dedupe_by_time(hits, limit)

    def moment(self, query: str, limit: int = TOP_K) -> list[dict]:
        """Pool passage scores up to the moment, then fuse with BM25 via RRF."""
        query_vector = self._query_vector(query)
        summary_scores = self.moment_vectors @ query_vector
        beat_scores = self.beat_vectors @ query_vector
        dense = np.zeros(len(self.moments), dtype=np.float32)
        for index, moment in enumerate(self.moments):
            left, right = moment["beat_start"], moment["beat_end"]
            passage = float(np.max(beat_scores[left:right])) if right > left else 0.0
            dense[index] = 0.75 * passage + 0.25 * float(summary_scores[index])
        dense_rank = [int(index) for index in np.argsort(-dense)]
        bm25_scores = self._moment_bm25.scores(query)
        bm25_rank = [int(index) for index in np.argsort(-bm25_scores)]
        fused = reciprocal_rank_fusion([dense_rank, bm25_rank])
        rrf_scores = np.zeros(len(self.moments), dtype=np.float32)
        for index, score in fused:
            rrf_scores[index] = score
        # Dense scores on one talk sit in a narrow band, so rank fusion alone
        # hides a rare exact phrase. Mix in the raw BM25 magnitude.
        combined = 0.45 * _scale(rrf_scores) + 0.55 * _scale(bm25_scores)
        order = np.argsort(-combined)
        hits = []
        for index in order:
            moment = self.moments[int(index)]
            hits.append(
                {
                    **{key: value for key, value in moment.items() if key not in {"beat_start", "beat_end"}},
                    "score": float(combined[index]),
                    "dense": float(dense[index]),
                    "bm25": float(bm25_scores[index]),
                    "kind": "moment",
                }
            )
        return hits[:limit]

    def _rebuild_cross_bm25(self) -> None:
        texts = [f"{moment['title']} {moment['text']}" for moment in self.moments]
        texts.extend(item["text"] for item in self.documents)
        self._cross_bm25 = BM25(texts)
        self._cross_count = len(texts)

    def add_documents(self, passages: list[dict], vectors: np.ndarray) -> None:
        """Persist new passages only after their vectors are in memory and on disk."""
        vectors = l2_normalize(np.asarray(vectors, dtype=np.float32))
        with self._lock:
            self.documents.extend(passages)
            self.doc_vectors = np.vstack([self.doc_vectors, vectors]) if len(self.doc_vectors) else vectors
            meta_path = INDEX_DIR / "meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["documents"] = self.documents
            np.save(INDEX_DIR / "docs.npy", self.doc_vectors)
            meta_path.write_text(json.dumps(meta), encoding="utf-8")
            self._rebuild_cross_bm25()

    def cross_source(self, query: str, limit: int = 6) -> list[dict]:
        """Retrieve video moments, paper pages, and deck slides from one index."""
        if self._cross_count == 0:
            return []
        query_vector = self._query_vector(query)
        moment_dense = self.moment_vectors @ query_vector if len(self.moments) else np.zeros(0, dtype=np.float32)
        doc_dense = self.doc_vectors @ query_vector if len(self.documents) else np.zeros(0, dtype=np.float32)
        dense = np.concatenate([moment_dense, doc_dense])
        bm25_scores = self._cross_bm25.scores(query)
        combined = 0.45 * _scale(dense) + 0.55 * _scale(bm25_scores)
        order = np.argsort(-combined)
        hits = []
        for index in order:
            score = float(combined[int(index)])
            if int(index) < len(self.moments):
                moment = self.moments[int(index)]
                hits.append({
                    **{key: value for key, value in moment.items() if key not in {"beat_start", "beat_end"}},
                    "score": score,
                    "kind": "video",
                    "source_title": self.video.get("title"),
                    "locator": {
                        "start_ms": int(moment["start"] * 1000),
                        "end_ms": int(moment["end"] * 1000),
                    },
                })
            else:
                document = self.documents[int(index) - len(self.moments)]
                hits.append({**document, "score": score})
        return _diversify(hits, limit)

    def ask(self, query: str) -> dict:
        query = query.strip()
        match_query = expand_query(query)
        baseline_hits = self.baseline(match_query)
        moment_hits = self.moment(match_query)
        baseline_answer, baseline_mode = answer(query, baseline_hits, match_query)
        moment_answer, moment_mode = answer(query, moment_hits, match_query)
        cross_hits = self.cross_source(match_query)
        return {
            "query": query,
            "video": self.video,
            "baseline": {
                "answer": baseline_answer,
                "mode": baseline_mode,
                "hits": [_public_hit(hit) for hit in baseline_hits],
            },
            "moment": {
                "answer": moment_answer,
                "mode": moment_mode,
                "hits": [_public_hit(hit) for hit in moment_hits],
            },
            "cross": {
                "answer": _cross_answer(query, cross_hits, match_query),
                "hits": [_public_hit(hit) for hit in cross_hits],
            },
        }


def _diversify(hits: list[dict], limit: int) -> list[dict]:
    """Keep the best passage of each kind, then fill by score."""
    chosen: list[dict] = []
    seen_ids: set[int] = set()
    seen_kinds: set[str] = set()
    for hit in hits:
        kind = hit["kind"]
        if kind in seen_kinds:
            continue
        chosen.append(hit)
        seen_ids.add(id(hit))
        seen_kinds.add(kind)
        if len(chosen) == limit:
            return chosen
    for hit in hits:
        if id(hit) in seen_ids:
            continue
        chosen.append(hit)
        seen_ids.add(id(hit))
        if len(chosen) == limit:
            break
    return chosen


def _cross_answer(query: str, hits: list[dict], match_query: str | None = None) -> str:
    if not hits:
        return "Nothing in the indexed sources matches this question."
    from app.answer import _window, strong_tokens

    terms = strong_tokens(match_query or query)
    lines = [f"Cross-source passages for “{query}”.", ""]
    for hit in hits[:3]:
        lines.append(f"{_locator_label(hit)}: {_window(hit['text'], terms)}")
    return "\n".join(lines)


def _locator_label(hit: dict) -> str:
    if hit.get("page"):
        return f"Paper p.{hit['page']}"
    if hit.get("slide"):
        return f"Slide {hit['slide']}"
    locator = hit.get("locator") or {}
    if "start_ms" in locator:
        from app.textutil import format_timestamp

        return f"Video {format_timestamp(locator['start_ms'] / 1000)}"
    from app.textutil import format_timestamp

    return f"Video {format_timestamp(hit.get('start') or 0)}"


def _public_hit(hit: dict) -> dict:
    return {
        "id": hit["id"],
        "kind": hit["kind"],
        "start": hit["start"],
        "end": hit["end"],
        "score": round(float(hit["score"]), 4),
        "title": hit.get("title"),
        "text": hit["text"],
        "sentences": hit.get("sentences") or [],
        "page": hit.get("page"),
        "slide": hit.get("slide"),
        "locator": hit.get("locator"),
        "source_title": hit.get("source_title"),
    }


@lru_cache(maxsize=1)
def get_index() -> SearchIndex:
    if not (INDEX_DIR / "meta.json").exists():
        raise FileNotFoundError("Index missing. Run `make index` first.")
    return SearchIndex(MiniLMEmbedder())


def video_payload() -> dict:
    index = get_index()
    return {
        "video": index.video,
        "samples": SAMPLE_QUERIES,
        "counts": {
            "chunks": len(index.chunks),
            "moments": len(index.moments),
            "documents": len(index.documents),
        },
    }
