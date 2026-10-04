import math
import re

import numpy as np

from app.textutil import STOPWORDS

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return [tok for tok in _TOKEN.findall(text.lower()) if tok not in STOPWORDS and len(tok) > 1]


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs = [_tokenize(doc) for doc in documents]
        self._avgdl = (sum(len(doc) for doc in self._docs) / len(self._docs)) if self._docs else 0.0
        df: dict[str, int] = {}
        for doc in self._docs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1
        self._df = df
        self._n = len(self._docs)

    def scores(self, query: str) -> np.ndarray:
        terms = _tokenize(query)
        scores = np.zeros(self._n, dtype=np.float32)
        if not terms or self._n == 0:
            return scores
        for index, doc in enumerate(self._docs):
            if not doc:
                continue
            counts: dict[str, int] = {}
            for term in doc:
                counts[term] = counts.get(term, 0) + 1
            dl = len(doc)
            score = 0.0
            for term in terms:
                freq = counts.get(term, 0)
                if freq == 0 or term not in self._df:
                    continue
                idf = math.log(1 + (self._n - self._df[term] + 0.5) / (self._df[term] + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1e-9))
                score += idf * (freq * (self.k1 + 1)) / denom
            scores[index] = score
        return scores


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)
