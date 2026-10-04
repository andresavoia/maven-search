import re

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE_END = re.compile(r'[.!?]["\']?$')

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "when", "while",
    "of", "to", "in", "on", "for", "with", "as", "at", "by", "from", "into",
    "is", "are", "was", "were", "be", "been", "being", "it", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "me", "my", "your",
    "our", "their", "what", "which", "who", "whom", "how", "why", "do", "does",
    "did", "doing", "have", "has", "had", "not", "no", "so", "than", "too",
    "very", "just", "about", "can", "could", "would", "should", "will", "there",
}


def tokens(text: str) -> list[str]:
    return [tok for tok in _TOKEN.findall(text.lower()) if tok not in STOPWORDS and len(tok) > 1]


def format_timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def sentences_from_cues(cues: list[dict]) -> list[dict]:
    """Group caption cues into sentences, falling back to length when punctuation is missing."""
    sentences: list[dict] = []
    buffer: list[str] = []
    start: float | None = None
    end = 0.0

    def flush() -> None:
        nonlocal buffer, start, end
        text = " ".join(part.strip() for part in buffer if part.strip())
        text = re.sub(r"\s+", " ", text).strip()
        if text and start is not None:
            sentences.append({"text": text, "start": start, "end": end})
        buffer = []
        start = None

    for cue in cues:
        piece = re.sub(r"\s+", " ", cue["text"]).strip()
        if not piece:
            continue
        if start is None:
            start = float(cue["start"])
        buffer.append(piece)
        end = float(cue["start"]) + float(cue.get("duration") or 0)
        joined = " ".join(buffer)
        if _SENTENCE_END.search(piece) or len(joined) >= 220:
            flush()
    flush()
    return sentences


_FILLER = {"um", "uh", "so", "like", "right", "okay", "ok", "well", "just", "kind", "sort", "and", "but", "the", "a"}


def title_from_sentences(sentences: list[dict]) -> str:
    if not sentences:
        return "Untitled moment"
    words = sentences[0]["text"].split()
    while words and words[0].lower().strip(".,") in _FILLER:
        words.pop(0)
    title = " ".join(words[:14]).strip(" .,")
    if len(title) < 12 and len(sentences) > 1:
        title = " ".join(sentences[1]["text"].split()[:14])
    if len(title) > 96:
        title = title[:93].rsplit(" ", 1)[0]
    return title or "Untitled moment"


def overlap_score(query: str, text: str) -> float:
    query_tokens = tokens(query)
    if not query_tokens:
        return 0.0
    text_counts: dict[str, int] = {}
    for tok in tokens(text):
        text_counts[tok] = text_counts.get(tok, 0) + 1
    hits = sum(text_counts.get(tok, 0) for tok in query_tokens)
    return hits / (len(query_tokens) + 0.5 * max(len(tokens(text)), 1) ** 0.5)
