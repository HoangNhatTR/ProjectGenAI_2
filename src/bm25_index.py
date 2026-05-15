"""BM25 keyword index, dùng song song với vector store (hybrid retrieval).

Lý do tồn tại: embedding (BGE-M3) tốt cho ngữ nghĩa nhưng yếu với các query
chứa danh từ riêng / số văn bản (vd: "Điều 5 Nghị định 100/2019"). BM25 khớp
chính xác từ-khoá → bù cho vector. Retriever sẽ trộn 2 kết quả qua RRF.

Persist: 1 file JSON ở `data/bm25/index.json` chứa danh sách chunk + token đã
tách. Khi load lại, dựng lại BM25Okapi từ tokenized corpus (rẻ).
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional

from .schemas import Chunk


_PUNCT_RE = re.compile(r"[^\w\s/\-]+", re.UNICODE)
_WS_RE = re.compile(r"\s+")


def tokenize(text: str) -> list[str]:
    """Tokenizer đơn giản cho BM25: lower + bỏ dấu câu, GIỮ chữ số / / - .

    Giữ '/' và '-' để các trích dẫn pháp luật như "100/2019/NĐ-CP", "1-3 tháng"
    không bị vỡ.
    """
    if not text:
        return []
    text = unicodedata.normalize("NFC", text).lower()
    text = _PUNCT_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text.split() if text else []


class BM25Index:
    """Persistent BM25 index. Lazy-load khi cần."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._bm25 = None
        self._chunks: list[Chunk] = []
        self._tokenized: list[list[str]] = []
        self._loaded = False

    # ===== Build / persistence =====

    def build(self, chunks: list[Chunk]) -> None:
        """(Re-)build index từ list chunks. Không tự save — gọi save() sau."""
        if not chunks:
            raise ValueError("Cần ít nhất 1 chunk để build BM25.")
        from rank_bm25 import BM25Okapi

        self._chunks = list(chunks)
        self._tokenized = [tokenize(c.text) for c in self._chunks]
        # rank_bm25 không chấp nhận tokenized rỗng — pad bằng placeholder
        safe_tokenized = [toks if toks else ["__empty__"] for toks in self._tokenized]
        self._bm25 = BM25Okapi(safe_tokenized)
        self._loaded = True

    def save(self) -> None:
        if not self._chunks:
            raise RuntimeError("Chưa có chunk nào trong index. Gọi build() trước.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "chunks": [c.model_dump(exclude_none=True) for c in self._chunks],
            "tokenized": self._tokenized,
        }
        fd, tmp_path = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self) -> None:
        """Đọc index từ disk. No-op nếu đã load."""
        if self._loaded:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"BM25 index không tồn tại: {self.path}")
        from rank_bm25 import BM25Okapi

        with self.path.open(encoding="utf-8") as f:
            data = json.load(f)
        self._chunks = [Chunk(**c) for c in data["chunks"]]
        self._tokenized = [list(toks) for toks in data["tokenized"]]
        safe_tokenized = [toks if toks else ["__empty__"] for toks in self._tokenized]
        self._bm25 = BM25Okapi(safe_tokenized)
        self._loaded = True

    # ===== Query =====

    def query(self, q: str, top_k: int) -> list[tuple[Chunk, float]]:
        """Trả về top_k chunks kèm BM25 score. Bỏ chunks có score <= 0."""
        if not q.strip():
            return []
        if not self._loaded:
            self.load()
        tokens = tokenize(q)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # argsort desc
        idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self._chunks[i], float(scores[i])) for i in idxs if scores[i] > 0]

    # ===== Introspection =====

    def is_available(self) -> bool:
        """True nếu đã load trong RAM hoặc có file trên disk."""
        return self._loaded or self.path.exists()

    def count(self) -> int:
        if not self._loaded and self.path.exists():
            self.load()
        return len(self._chunks)
