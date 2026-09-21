"""Embedding model wrapper.

Uses sentence-transformers/all-MiniLM-L6-v2 (free, local) by default. If the
model can't be loaded (no internet on first run, package not installed, or
tests running hermetically), falls back to a deterministic hashing-trick
embedding so the rest of the pipeline stays fully testable offline. The
fallback is announced via `.backend` so callers/logs can tell which is active.
"""
import hashlib
import re
from functools import lru_cache

import numpy as np

from app.config import get_settings

_HASH_DIM = 384  # matches all-MiniLM-L6-v2 output dim, for interface parity


def _hash_embed(text: str, dim: int = _HASH_DIM) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    for tok in tokens:
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


class EmbeddingModel:
    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model
        self._model = None
        self.backend = "hash-fallback"
        self._try_load_real_model()

    def _try_load_real_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            self.backend = "sentence-transformers"
        except Exception:
            self._model = None
            self.backend = "hash-fallback"

    def encode(self, texts: list[str]) -> list[np.ndarray]:
        if self._model is not None:
            embeddings = self._model.encode(texts, normalize_embeddings=True)
            return [np.asarray(e, dtype=np.float32) for e in embeddings]
        return [_hash_embed(t) for t in texts]

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


@lru_cache
def get_embedding_model() -> EmbeddingModel:
    return EmbeddingModel()
