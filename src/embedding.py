from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .schemas import Chunk


class Embedder:
    """Wrapper quanh sentence-transformers.

    Mặc định dùng AITeamVN/Vietnamese_Embedding (fine-tune từ BGE-M3 cho tiếng Việt).
    Embedding được L2-normalize → dot product = cosine similarity.

    Thiết kế để dễ swap sang OpenAI / Voyage embeddings sau này — chỉ cần
    giữ chữ ký encode(texts) -> list[list[float]].
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: int = 16,
        max_seq_length: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self._model: Optional[SentenceTransformer] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        self._model = SentenceTransformer(self.model_name, device=self.device)
        self._model.max_seq_length = self.max_seq_length

    @property
    def dim(self) -> int:
        self._load()
        return self._model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        self._load()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 16,
        )
        return embeddings.tolist()

    def encode_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return self.encode([c.text for c in chunks])


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vector. Giả định đã normalize → dot product."""
    return float(np.dot(np.array(a), np.array(b)))
