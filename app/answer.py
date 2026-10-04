import os
import re

import httpx

from app.config import OPENAI_API_KEY, OPENAI_MODEL
from app.textutil import format_timestamp, overlap_score, tokens


_WEAK = {
    "after", "before", "work", "works", "working", "using", "use", "used",
    "like", "just", "really", "something", "thing", "things", "about", "into",
}


def strong_tokens(query: str) -> list[str]:
    seen: list[str] = []
    for tok in tokens(query):
        if tok in _WEAK or len(tok) < 4 or tok in seen:
            continue
        seen.append(tok)
    return seen


def _lead(query: str, hits: list[dict]) -> str:
    if not hits:
        return "The transcript does not contain a passage that answers this."
    start = format_timestamp(hits[0]["start"])
    end = format_timestamp(hits[0]["end"])
    label = hits[0].get("title")
    where = f"{start}–{end}"
    if label:
        return f"Best match for “{query}” is {where}: {label}."
    return f"Best match for “{query}” is {where}."


def _window(text: str, terms: list[str], size: int = 320) -> str:
    lowered = text.lower()
    positions = [lowered.find(term) for term in terms if term in lowered]
    positions = [pos for pos in positions if pos >= 0]
    if not positions:
        return text[:size].strip()
    anchor = min(positions)
    left = max(0, anchor - 90)
    snippet = text[left : left + size].strip()
    if left > 0:
        snippet = "…" + snippet
    if left + size < len(text):
        snippet = snippet + "…"
    return snippet


def _term_in(text: str, term: str) -> bool:
    return re.search(rf"\b{re.escape(term)}s?\b", text.lower()) is not None


def extractive_answer(query: str, hits: list[dict], match_query: str | None = None) -> str:
    """Grounded answer built only from retrieved sentences. No invented facts."""
    if not hits:
        return "Nothing in the indexed transcript matches this question."

    needed = strong_tokens(match_query or query)
    weights = {}
    for term in needed:
        docs = sum(1 for hit in hits if _term_in(hit["text"], term))
        weights[term] = 1.0 / (docs + 0.25)
    candidates: list[dict] = []
    for hit_index, hit in enumerate(hits[:1]):
        for sentence in hit.get("sentences") or []:
            present = [term for term in needed if _term_in(sentence["text"], term)]
            weight = sum(weights[term] for term in present)
            if needed and weight <= 0:
                continue
            lexical = overlap_score(match_query or query, sentence["text"])
            if weight <= 0 and lexical <= 0:
                continue
            candidates.append({
                **sentence,
                "score": (weight + lexical) / (1 + hit_index * 0.2),
            })
    candidates.sort(key=lambda item: item["score"], reverse=True)
    chosen: list[dict] = []
    seen: set[str] = set()
    for sentence in candidates:
        key = sentence["text"].lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append(sentence)
        if len(chosen) == 3:
            break
    if not chosen:
        top = hits[0]
        chosen = [{
            "text": _window(top["text"], needed),
            "start": top["start"],
            "end": top["end"],
        }]
    chosen.sort(key=lambda item: item["start"])
    lines = [_lead(query, hits), ""]
    for sentence in chosen:
        lines.append(f"At {format_timestamp(sentence['start'])}: {sentence['text']}")
    return "\n".join(lines)


def _context_block(hits: list[dict]) -> str:
    blocks = []
    for hit in hits:
        label = hit.get("title") or "chunk"
        blocks.append(
            f"[{format_timestamp(hit['start'])}-{format_timestamp(hit['end'])}] {label}\n{hit['text']}"
        )
    return "\n\n".join(blocks)


def maybe_llm_answer(query: str, hits: list[dict]) -> str | None:
    api_key = os.environ.get("OPENAI_API_KEY", OPENAI_API_KEY)
    if not api_key or not hits:
        return None
    system = (
        "Answer only from the transcript excerpts. "
        "Cite timestamps that appear in the excerpts. "
        "If the excerpts do not contain the answer, say so. "
        "Do not invent a timestamp, a speaker, or a fact."
    )
    payload = {
        "model": os.environ.get("OPENAI_MODEL", OPENAI_MODEL),
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Question: {query}\n\nExcerpts:\n{_context_block(hits)}",
            },
        ],
    }
    response = httpx.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def answer(query: str, hits: list[dict], match_query: str | None = None) -> tuple[str, str]:
    generated = maybe_llm_answer(query, hits)
    if generated:
        return generated, "openai"
    return extractive_answer(query, hits, match_query), "extractive"
