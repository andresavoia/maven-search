import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-9)
    return matrix / norms


class HashEmbedder:
    """Deterministic bag-of-words embedder used by tests. Not the production model."""

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in _TOKEN.findall(text.lower()):
                digest = hashlib.sha256(token.encode()).digest()
                index = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vectors[row, index] += sign
        return l2_normalize(vectors)


class MiniLMEmbedder:
    """Local sentence-transformer. Downloads all-MiniLM-L6-v2 on first use."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        name = "sentence-transformers/all-MiniLM-L6-v2"
        try:
            self._model = SentenceTransformer(name, local_files_only=True)
        except Exception:
            self._model = SentenceTransformer(name)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vectors = self._model.encode(
            texts,
            batch_size=64,
            show_progress_bar=len(texts) > 32,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype=np.float32)
