from __future__ import annotations

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .schemas import Chunk


class Embedder:
    """Wrapper quanh sentence-transformers (mặc định BAAI/bge-m3).

    Embedding được L2-normalize → dot product = cosine similarity.

    Tự động tối ưu theo phần cứng:
      - device:     None → auto (cuda nếu có, ngược lại cpu)
      - use_fp16:   None → auto (bật trên cuda — T4/A100 có Tensor Cores,
                    nhanh ~2×, chất lượng gần như không đổi; tắt trên cpu)
      - batch_size: None → auto (128 cho A100/L4/H100, 64 cho GPU khác, 16 cpu)

    Thiết kế để dễ swap sang OpenAI / Voyage embeddings sau này — chỉ cần
    giữ chữ ký encode(texts) -> list[list[float]].
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_seq_length: int = 1024,
        use_fp16: Optional[bool] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.use_fp16 = use_fp16
        self._model: Optional[SentenceTransformer] = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        use_fp16 = self.use_fp16 if self.use_fp16 is not None else device == "cuda"
        model_kwargs = {"torch_dtype": torch.float16} if use_fp16 else {}

        self._model = SentenceTransformer(
            self.model_name, device=device, model_kwargs=model_kwargs,
        )
        self._model.max_seq_length = self.max_seq_length

        if self.batch_size is None:
            if device == "cuda":
                gpu = torch.cuda.get_device_name(0)
                self.batch_size = 128 if any(x in gpu for x in ("A100", "L4", "H100")) else 64
            else:
                self.batch_size = 16

        print(
            f"  Embedder: device={device}, fp16={'on' if use_fp16 else 'off'}, "
            f"batch_size={self.batch_size}, max_seq={self.max_seq_length}"
        )

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
        # FP16 → cast về float32 trước khi lưu (Chroma/cosine ổn định hơn)
        return embeddings.astype(np.float32).tolist()

    def encode_chunks(self, chunks: list[Chunk]) -> list[list[float]]:
        return self.encode([c.text for c in chunks])


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vector. Giả định đã normalize → dot product."""
    return float(np.dot(np.array(a), np.array(b)))
