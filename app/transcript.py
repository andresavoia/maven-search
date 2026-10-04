import json

import httpx

from app.config import TRANSCRIPT_PATH, VIDEO_AUTHOR, VIDEO_ID, VIDEO_TITLE, VIDEO_URL


def _fetch_cues(video_id: str) -> list[dict]:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    cues = []
    for snippet in fetched:
        text = getattr(snippet, "text", None)
        start = getattr(snippet, "start", None)
        duration = getattr(snippet, "duration", None)
        if text is None and isinstance(snippet, dict):
            text = snippet.get("text")
            start = snippet.get("start")
            duration = snippet.get("duration")
        cues.append({"text": text or "", "start": float(start or 0), "duration": float(duration or 0)})
    return cues


def _fetch_oembed() -> dict:
    response = httpx.get(
        "https://www.youtube.com/oembed",
        params={"url": VIDEO_URL, "format": "json"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return {"title": payload.get("title") or VIDEO_TITLE, "author": payload.get("author_name") or VIDEO_AUTHOR}


def load_or_fetch() -> dict:
    if TRANSCRIPT_PATH.exists():
        return json.loads(TRANSCRIPT_PATH.read_text(encoding="utf-8"))
    meta = _fetch_oembed()
    document = {
        "video_id": VIDEO_ID,
        "url": VIDEO_URL,
        "title": meta["title"],
        "author": meta["author"],
        "cues": _fetch_cues(VIDEO_ID),
    }
    TRANSCRIPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRANSCRIPT_PATH.write_text(json.dumps(document), encoding="utf-8")
    return document
