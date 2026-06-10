"""Retrieval + Rerank cache — LRU với TTL.

Tránh chạy lại embedding + BM25 + CrossEncoder cho cùng query.

Cách dùng:
    from src.cache import RetrievalCache
    cache = RetrievalCache(maxsize=512, ttl=3600)
    results = cache.get(query, top_k, min_score)
    if results is None:
        results = retriever.retrieve(...); results = rerank(...)
        cache.put(query, top_k, min_score, results)
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import Optional

from .schemas import RetrievedChunk


def _make_key(query: str, top_k: int, min_score: Optional[float]) -> str:
    raw = f"{query.strip().lower()}|{top_k}|{min_score}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class RetrievalCache:
    """Thread-safe LRU cache với TTL cho kết quả retrieve+rerank."""

    def __init__(self, maxsize: int = 512, ttl: float = 3600.0):
        self.maxsize = maxsize
        self.ttl = ttl
        self._lock = threading.Lock()
        # key → (timestamp, results)
        self._store: OrderedDict[str, tuple[float, list[RetrievedChunk]]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(
        self,
        query: str,
        top_k: int,
        min_score: Optional[float],
    ) -> Optional[list[RetrievedChunk]]:
        key = _make_key(query, top_k, min_score)
        with self._lock:
            if key not in self._store:
                self.misses += 1
                return None
            ts, results = self._store[key]
            if time.time() - ts > self.ttl:
                del self._store[key]
                self.misses += 1
                return None
            # LRU: di chuyển lên đầu
            self._store.move_to_end(key)
            self.hits += 1
            return results

    def put(
        self,
        query: str,
        top_k: int,
        min_score: Optional[float],
        results: list[RetrievedChunk],
    ) -> None:
        key = _make_key(query, top_k, min_score)
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (time.time(), results)
            # Evict entry cũ nhất nếu vượt maxsize
            while len(self._store) > self.maxsize:
                self._store.popitem(last=False)

    def invalidate(self) -> None:
        with self._lock:
            self._store.clear()
            self.hits = 0
            self.misses = 0

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def stats_str(self) -> str:
        total = self.hits + self.misses
        rate = self.hits / total * 100 if total else 0
        return f"size={self.size}/{self.maxsize} hits={self.hits} miss={self.misses} rate={rate:.0f}%"
