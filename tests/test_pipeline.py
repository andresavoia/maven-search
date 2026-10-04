import json

import numpy as np

from app.answer import extractive_answer
from app.bm25 import BM25, reciprocal_rank_fusion
from app.chunking import fixed_chunks, prepare_cues
from app.embeddings import HashEmbedder
from app.moments import make_beats, segment_moments
from app.search import SearchIndex
from app.textutil import sentences_from_cues, tokens


def _cue(start: float, text: str, duration: float = 4.0) -> dict:
    return {"start": start, "duration": duration, "text": text}


def _talk() -> list[dict]:
    security = [
        "Prompt injection hides instructions inside untrusted data.",
        "A jailbreak tries to push the model past its safety training.",
        "Data poisoning can change behavior if bad documents enter the training set.",
    ]
    training = [
        "Pretraining predicts the next token on a huge pile of internet text.",
        "Finetuning teaches the model to follow instructions from curated examples.",
        "RLHF uses human preferences to rank answers and update the model.",
    ]
    operating = [
        "An LLM operating system coordinates tools memory and a user interface.",
        "The model becomes the kernel that calls browsers calculators and code.",
    ]
    cues = []
    clock = 0.0
    for sentence in training * 6:
        cues.append(_cue(clock, sentence, 5))
        clock += 5
    for sentence in security * 8:
        cues.append(_cue(clock, sentence, 5))
        clock += 5
    for sentence in operating * 6:
        cues.append(_cue(clock, sentence, 5))
        clock += 5
    return cues


def test_tokens_drop_stopwords():
    assert "the" not in tokens("What is the model")
    assert "model" in tokens("What is the model")


def test_fixed_chunks_overlap_and_cover_the_start():
    cues = prepare_cues(_talk())
    chunks = fixed_chunks(cues, words_per_chunk=40, overlap_words=10)
    assert chunks
    assert chunks[0]["start"] == cues[0]["start"]
    assert all(chunk["end"] >= chunk["start"] for chunk in chunks)
    assert chunks[1]["start"] < chunks[0]["end"]


def test_moments_follow_topic_shifts():
    cues = prepare_cues(_talk())
    beats = make_beats(cues, target_seconds=20)
    embedder = HashEmbedder()
    vectors = embedder.embed([beat["text"] for beat in beats])
    moments = segment_moments(beats, vectors)
    assert len(moments) >= 2
    blob = " ".join(moment["text"] for moment in moments)
    assert "jailbreak" in blob
    assert "RLHF" in blob
    assert moments[0]["start"] <= moments[-1]["start"]
    for moment in moments:
        assert moment["end"] > moment["start"]
        assert moment["title"]


def test_bm25_prefers_the_document_that_contains_the_term():
    ranker = BM25(["apples and oranges", "rlhf uses human preferences", "the weather is rainy"])
    scores = ranker.scores("how does rlhf work")
    assert int(np.argmax(scores)) == 1


def test_rrf_promotes_a_document_ranked_well_in_both_lists():
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 2, 0]])
    assert fused[0][0] == 1


def test_extractive_answer_cites_a_real_timestamp():
    hits = [
        {
            "start": 125,
            "end": 180,
            "text": "RLHF uses human preferences to rank answers.",
            "sentences": [{"text": "RLHF uses human preferences to rank answers.", "start": 130, "end": 140}],
        }
    ]
    text = extractive_answer("How does RLHF work?", hits)
    assert "2:10" in text
    assert "RLHF" in text


def test_search_index_returns_security_moment(tmp_path, monkeypatch):
    cues = prepare_cues(_talk())
    beats = make_beats(cues, target_seconds=20)
    embedder = HashEmbedder()
    beat_vectors = embedder.embed([beat["text"] for beat in beats])
    moments = segment_moments(beats, beat_vectors)
    chunks = fixed_chunks(cues, words_per_chunk=40, overlap_words=8)
    chunk_vectors = embedder.embed([chunk["text"] for chunk in chunks])
    moment_vectors = embedder.embed([f"{moment['title']}. {moment['text'][:400]}" for moment in moments])

    index_dir = tmp_path / "index"
    index_dir.mkdir()
    meta = {
        "video": {"id": "test", "title": "fixture", "url": "https://example.com", "author": "test", "why": "test", "duration": 100, "cue_count": len(cues)},
        "chunks": chunks,
        "moments": moments,
    }
    (index_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.save(index_dir / "chunks.npy", chunk_vectors)
    np.save(index_dir / "moments.npy", moment_vectors)
    np.save(index_dir / "beats.npy", beat_vectors)

    monkeypatch.setattr("app.search.INDEX_DIR", index_dir)
    engine = SearchIndex(embedder)
    result = engine.ask("What is a jailbreak attack on a language model?")
    moment_text = " ".join(hit["text"] for hit in result["moment"]["hits"])
    assert "jailbreak" in moment_text.lower() or "prompt injection" in moment_text.lower()
    assert result["moment"]["hits"][0]["end"] > result["moment"]["hits"][0]["start"]
    assert result["baseline"]["answer"]
    assert result["moment"]["mode"] == "extractive"
