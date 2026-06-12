"""ChromaFTSIndex — nhánh lexical search dùng FTS5 có sẵn trong chroma.sqlite3.

Thay thế BM25Index (rank_bm25) khi không có/không load nổi index.json:
  - index.json của corpus 4.9M chunks nặng 9.2GB, load bằng rank_bm25
    ngốn ~25GB+ RAM → không khả thi trên máy thường lẫn Colab chuẩn
  - Chroma đã build sẵn FTS5 (tokenize='trigram') trên text mọi chunk
    trong chroma.sqlite3 → query trực tiếp, ZERO RAM thêm, zero build

Interface giống BM25Index: is_available() + query(q, top_k) -> [(Chunk, score)].
Tokenizer trigram: token < 3 ký tự không match được → build match expression
từ các CỤM 2 TỪ liên tiếp (vd "đèn đỏ", "vượt đèn") + từ đơn dài ≥ 4 ký tự.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from loguru import logger

from .schemas import Chunk
from .vectorstore import _chroma_to_chunk

_WORD_RE = re.compile(r"[\wÀ-ỹ]+", re.UNICODE)


def _build_match(query: str) -> str:
    """Câu hỏi tự nhiên → biểu thức FTS5 MATCH.

    'xe máy vượt đèn đỏ phạt bao nhiêu' →
    '"xe máy" OR "máy vượt" OR "vượt đèn" OR "đèn đỏ" OR ... OR "phạt"'
    """
    words = _WORD_RE.findall(query.lower())
    parts: list[str] = []
    # Cụm 2 từ liên tiếp — tín hiệu lexical mạnh nhất với trigram
    for a, b in zip(words, words[1:]):
        parts.append(f'"{a} {b}"')
    # Từ đơn đủ dài (trigram cần ≥3 ký tự; lấy ≥4 để bớt nhiễu)
    parts.extend(f'"{w}"' for w in words if len(w) >= 4)
    # Dedup giữ thứ tự
    seen: set[str] = set()
    uniq = [p for p in parts if not (p in seen or seen.add(p))]
    return " OR ".join(uniq)


class ChromaFTSIndex:
    """Lexical search trên bảng embedding_fulltext_search của Chroma."""

    def __init__(self, vectorstore_dir: Path):
        self.db_path = Path(vectorstore_dir) / "chroma.sqlite3"
        self._available: bool | None = None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.db_path.as_posix()}?mode=ro", uri=True)

    def is_available(self) -> bool:
        if self._available is None:
            try:
                con = self._connect()
                try:
                    row = con.execute(
                        "SELECT name FROM sqlite_master "
                        "WHERE name = 'embedding_fulltext_search'"
                    ).fetchone()
                finally:
                    con.close()
                self._available = row is not None
            except Exception as exc:
                logger.warning(f"ChromaFTS không khả dụng: {exc}")
                self._available = False
        return self._available

    def query(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Trả [(Chunk, score)] xếp hạng tốt nhất trước (score càng cao càng tốt)."""
        match = _build_match(query)
        if not match or not self.is_available():
            return []
        try:
            con = self._connect()
            try:
                hits = con.execute(
                    "SELECT e.id, e.embedding_id, bm25(embedding_fulltext_search) AS s "
                    "FROM embedding_fulltext_search f "
                    "JOIN embeddings e ON e.id = f.rowid "
                    "WHERE embedding_fulltext_search MATCH ? "
                    "ORDER BY s LIMIT ?",
                    (match, top_k),
                ).fetchall()
                if not hits:
                    return []

                # Lấy text + metadata cho các id trúng (1 query pivot)
                ids = [h[0] for h in hits]
                ph = ",".join("?" * len(ids))
                meta_rows = con.execute(
                    f"SELECT id, key, string_value FROM embedding_metadata "
                    f"WHERE id IN ({ph})",
                    ids,
                ).fetchall()
            finally:
                con.close()

            by_id: dict[int, dict] = {i: {} for i in ids}
            for rid, key, sval in meta_rows:
                by_id[rid][key] = sval

            results: list[tuple[Chunk, float]] = []
            for rid, embedding_id, score in hits:
                meta = by_id.get(rid, {})
                text = meta.pop("chroma:document", "") or ""
                chunk = _chroma_to_chunk(embedding_id, text, meta)
                # bm25() của FTS5: càng âm càng tốt → đổi dấu cho descending
                results.append((chunk, -float(score)))
            return results
        except Exception as exc:
            logger.warning(f"ChromaFTS query lỗi: {exc}")
            return []
