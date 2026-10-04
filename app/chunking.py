import re

from app.config import CHUNK_OVERLAP_WORDS, WORDS_PER_CHUNK
from app.textutil import sentences_from_cues


def clean_cue_text(text: str) -> str:
    text = text.replace("\n", " ")
    text = re.sub(r"\[.*?\]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def prepare_cues(raw_cues: list[dict]) -> list[dict]:
    cues: list[dict] = []
    for raw in raw_cues:
        text = clean_cue_text(str(raw.get("text", "")))
        if not text:
            continue
        start = float(raw["start"])
        duration = float(raw.get("duration") or 0)
        cues.append({"start": start, "duration": duration, "text": text})
    cues.sort(key=lambda cue: cue["start"])
    return cues


def fixed_chunks(
    cues: list[dict],
    words_per_chunk: int = WORDS_PER_CHUNK,
    overlap_words: int = CHUNK_OVERLAP_WORDS,
) -> list[dict]:
    """Split the transcript into overlapping word windows. Boundaries ignore topic shifts."""
    words: list[tuple[str, float, float]] = []
    for cue in cues:
        end = cue["start"] + cue["duration"]
        for word in cue["text"].split():
            words.append((word, cue["start"], end))
    if not words:
        return []

    step = max(1, words_per_chunk - overlap_words)
    chunks: list[dict] = []
    start_index = 0
    while start_index < len(words):
        window = words[start_index : start_index + words_per_chunk]
        if len(window) < 30 and chunks:
            break
        text = " ".join(word for word, _, _ in window)
        covered = [
            cue
            for cue in cues
            if cue["start"] <= window[-1][2] and cue["start"] + cue["duration"] >= window[0][1]
        ]
        chunks.append(
            {
                "id": f"chunk-{len(chunks)}",
                "start": window[0][1],
                "end": window[-1][2],
                "text": text,
                "sentences": sentences_from_cues(covered) if covered else [{"text": text, "start": window[0][1], "end": window[-1][2]}],
            }
        )
        if start_index + words_per_chunk >= len(words):
            break
        start_index += step
    return chunks
