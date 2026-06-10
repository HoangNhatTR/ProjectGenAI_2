"""SQLite-backed store cho parent chunks.

Parent chunk = toàn bộ text của 1 Điều (tối đa MAX_PARENT_CHARS ký tự).
Child chunks trong Chroma có parent_id trỏ vào đây.

Sau khi retrieve child chunks → gọi get() để lấy text đầy đủ hơn cho Generator.
Backward compatible: chunk cũ không có parent_id → không được expand.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


class ParentStore:
    """Key-value store (SQLite) ánh xạ parent_id → full text của Điều."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = str(db_path)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS parents "
                "(id TEXT PRIMARY KEY, text TEXT NOT NULL)"
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_id ON parents(id)")
            conn.commit()

    # ── Write ──────────────────────────────────────────────────────────────────

    def add_batch(self, items: list[tuple[str, str]]) -> None:
        """Lưu nhiều (parent_id, text) trong 1 transaction."""
        if not items:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO parents VALUES (?, ?)", items
            )
            conn.commit()

    # ── Read ───────────────────────────────────────────────────────────────────

    def get(self, parent_id: str) -> Optional[str]:
        """Trả về text của parent chunk, None nếu không tìm thấy."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT text FROM parents WHERE id = ?", (parent_id,)
            ).fetchone()
        return row[0] if row else None

    def get_batch(self, parent_ids: list[str]) -> dict[str, str]:
        """Lấy nhiều parent cùng lúc. Trả về dict id → text."""
        if not parent_ids:
            return {}
        placeholders = ",".join("?" * len(parent_ids))
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                f"SELECT id, text FROM parents WHERE id IN ({placeholders})",
                parent_ids,
            ).fetchall()
        return {row[0]: row[1] for row in rows}

    def count(self) -> int:
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute("SELECT COUNT(*) FROM parents").fetchone()[0]

    def reset(self) -> None:
        """Xóa toàn bộ store (dùng khi reindex)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM parents")
            conn.commit()
