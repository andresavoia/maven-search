from app.eval_retrieval import covers_gold, first_hit_rank, reciprocal_rank, summarize


def test_overlap_requires_a_shared_interval():
    hit = {"start": 100, "end": 140}
    assert covers_gold(hit, [(120, 180)])
    assert not covers_gold(hit, [(140, 180)])
    assert not covers_gold(hit, [(10, 20)])


def test_rank_is_the_first_overlapping_hit():
    hits = [
        {"start": 0, "end": 30},
        {"start": 100, "end": 160},
        {"start": 200, "end": 240},
    ]
    assert first_hit_rank(hits, [(150, 170)]) == 2
    assert reciprocal_rank(hits, [(150, 170)]) == 0.5
    assert first_hit_rank(hits, [(900, 950)]) is None
    assert reciprocal_rank(hits, [(900, 950)]) == 0.0


def test_summary_averages_hit_rates():
    rows = [
        {"baseline": {"hit_at_1": True, "hit_at_4": True, "mrr": 1.0}},
        {"baseline": {"hit_at_1": False, "hit_at_4": True, "mrr": 0.5}},
    ]
    stats = summarize(rows, "baseline")
    assert stats["hit_at_1"] == 0.5
    assert stats["hit_at_4"] == 1.0
    assert stats["mrr"] == 0.75
    assert stats["n"] == 2
