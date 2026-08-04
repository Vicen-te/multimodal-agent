"""Sentence-Transformers embedder for the RAG retriever."""

from __future__ import annotations

import numpy as np


class SentenceTransformerEmbedder:
    """Wrap a Sentence-Transformers model behind a small ``embed`` interface.

    The model is loaded lazily on first use so that importing this module (and the
    package as a whole) does not pull torch into memory until embeddings are
    actually needed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            # Pinned to CPU: the model is small enough that it never needs a GPU,
            # and hosts like ZeroGPU Spaces only expose CUDA inside decorated
            # functions, where a default "auto" device would fail at startup.
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def embed(self, texts: list[str]) -> np.ndarray:
        model = self._ensure_model()
        vectors = model.encode(list(texts), normalize_embeddings=True)
        return np.asarray(vectors, dtype=np.float32)
