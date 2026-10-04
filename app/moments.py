import numpy as np

from app.config import BEAT_SECONDS, MAX_MOMENT_SECONDS, MIN_MOMENT_SECONDS
from app.textutil import sentences_from_cues, title_from_sentences


def make_beats(cues: list[dict], target_seconds: float = BEAT_SECONDS) -> list[dict]:
    """Pack caption cues into short beats. A moment is a run of related beats."""
    beats: list[dict] = []
    bucket: list[dict] = []

    def flush() -> None:
        if not bucket:
            return
        text = " ".join(cue["text"] for cue in bucket)
        beats.append(
            {
                "start": bucket[0]["start"],
                "end": bucket[-1]["start"] + bucket[-1]["duration"],
                "text": text,
                "cues": list(bucket),
            }
        )
        bucket.clear()

    for cue in cues:
        bucket.append(cue)
        start = bucket[0]["start"]
        end = cue["start"] + cue["duration"]
        if end - start >= target_seconds:
            flush()
    flush()
    return beats


def _adjacent_similarity(embeddings: np.ndarray) -> np.ndarray:
    if len(embeddings) < 2:
        return np.zeros(0, dtype=np.float32)
    return np.sum(embeddings[:-1] * embeddings[1:], axis=1)


def _boundary_indexes(similarities: np.ndarray) -> list[int]:
    """Return beat indexes where a new moment should start."""
    if len(similarities) == 0:
        return []
    threshold = float(np.median(similarities) - 0.45 * np.std(similarities))
    cuts: list[int] = []
    for index, similarity in enumerate(similarities):
        left = similarities[index - 1] if index > 0 else similarity + 1
        right = similarities[index + 1] if index + 1 < len(similarities) else similarity + 1
        if similarity <= threshold and similarity <= left and similarity <= right:
            cuts.append(index + 1)
    return cuts


def _ranges_from_cuts(count: int, cuts: list[int]) -> list[tuple[int, int]]:
    points = [0, *cuts, count]
    ranges: list[tuple[int, int]] = []
    for left, right in zip(points, points[1:]):
        if right > left:
            ranges.append((left, right))
    return ranges or [(0, count)]


def _duration(beats: list[dict], start: int, end: int) -> float:
    return beats[end - 1]["end"] - beats[start]["start"]


def _merge_short(beats: list[dict], ranges: list[tuple[int, int]], similarities: np.ndarray) -> list[tuple[int, int]]:
    changed = True
    while changed and len(ranges) > 1:
        changed = False
        for index, (left, right) in enumerate(ranges):
            if _duration(beats, left, right) >= MIN_MOMENT_SECONDS:
                continue
            if index == 0:
                neighbor = 1
            elif index == len(ranges) - 1:
                neighbor = index - 1
            else:
                left_sim = float(similarities[ranges[index - 1][1] - 1]) if ranges[index - 1][1] - 1 < len(similarities) else 0
                right_sim = float(similarities[right - 1]) if right - 1 < len(similarities) else 0
                neighbor = index - 1 if left_sim >= right_sim else index + 1
            other_left, other_right = ranges[neighbor]
            merged = (min(left, other_left), max(right, other_right))
            keep = index if neighbor > index else neighbor
            drop = neighbor if neighbor > index else index
            ranges[keep] = merged
            del ranges[drop]
            changed = True
            break
    return ranges


def _split_long(beats: list[dict], ranges: list[tuple[int, int]], similarities: np.ndarray) -> list[tuple[int, int]]:
    """Split a long moment only where both sides stay long enough to be a moment."""
    result: list[tuple[int, int]] = []
    for left, right in ranges:
        if _duration(beats, left, right) <= MAX_MOMENT_SECONDS or right - left < 3:
            result.append((left, right))
            continue
        candidates: list[tuple[float, int]] = []
        for split_at in range(left + 1, right):
            if _duration(beats, left, split_at) < MIN_MOMENT_SECONDS:
                continue
            if _duration(beats, split_at, right) < MIN_MOMENT_SECONDS:
                continue
            candidates.append((float(similarities[split_at - 1]), split_at))
        if not candidates:
            result.append((left, right))
            continue
        _, split_at = min(candidates)
        result.extend(_split_long(beats, [(left, split_at), (split_at, right)], similarities))
    return result


def segment_moments(beats: list[dict], embeddings: np.ndarray) -> list[dict]:
    """Cut the talk where adjacent beats stop talking about the same thing."""
    if not beats:
        return []
    if len(beats) != len(embeddings):
        raise ValueError("Each beat needs one embedding")
    similarities = _adjacent_similarity(embeddings)
    ranges = _ranges_from_cuts(len(beats), _boundary_indexes(similarities))
    ranges = _merge_short(beats, ranges, similarities)
    ranges = _split_long(beats, ranges, similarities)
    ranges = _merge_short(beats, ranges, similarities)

    moments: list[dict] = []
    for left, right in ranges:
        context_left = max(0, left - 1)
        selected = beats[context_left:right]
        cues = [cue for beat in selected for cue in beat["cues"]]
        sentences = sentences_from_cues(cues)
        text = " ".join(beat["text"] for beat in selected)
        end = beats[right]["start"] if right < len(beats) else beats[right - 1]["end"]
        start = selected[0]["start"]
        if end <= start:
            end = beats[right - 1]["end"]
        moments.append(
            {
                "id": f"moment-{len(moments)}",
                "start": start,
                "end": end,
                "title": title_from_sentences(sentences_from_cues([cue for beat in beats[left:right] for cue in beat["cues"]])),
                "text": text,
                "sentences": sentences,
                "beat_start": left,
                "beat_end": right,
            }
        )
    return moments
