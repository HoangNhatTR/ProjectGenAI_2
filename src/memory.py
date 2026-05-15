"""Long-term memory bền vững xuyên session.

Khác với session history (mỗi phiên 1 list message), memory là tập fact ngắn
về người dùng, tồn tại mãi và được inject vào system prompt mỗi lượt.
"""
from __future__ import annotations

import json
import os
import tempfile
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _make_id() -> str:
    return "m_" + uuid.uuid4().hex[:8]


@dataclass
class MemoryFact:
    id: str
    content: str
    created_at: str
    source_session: Optional[str] = None

    @classmethod
    def new(cls, content: str, source_session: Optional[str] = None) -> "MemoryFact":
        return cls(
            id=_make_id(),
            content=content.strip(),
            created_at=_now_iso(),
            source_session=source_session,
        )

    @classmethod
    def from_json(cls, data: dict) -> "MemoryFact":
        return cls(
            id=data["id"],
            content=data["content"],
            created_at=data.get("created_at", _now_iso()),
            source_session=data.get("source_session"),
        )

    def to_json(self) -> dict:
        return asdict(self)


@dataclass
class MemoryStore:
    """JSON-file store cho memory. Tất cả fact trong 1 file `memory.json`."""

    path: Path
    _facts: list[MemoryFact] = field(default_factory=list, init=False)
    _loaded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            try:
                with self.path.open(encoding="utf-8") as f:
                    raw = json.load(f)
                self._facts = [MemoryFact.from_json(d) for d in raw]
            except (json.JSONDecodeError, OSError, KeyError):
                self._facts = []
        self._loaded = True

    def _save(self) -> None:
        fd, tmp_path = tempfile.mkstemp(dir=self.path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump([m.to_json() for m in self._facts], f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, self.path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def all(self) -> list[MemoryFact]:
        return list(self._facts)

    def add(self, content: str, source_session: Optional[str] = None) -> Optional[MemoryFact]:
        """Thêm fact mới. Trả None nếu đã có fact tương tự (case-insensitive)."""
        content = content.strip()
        if not content:
            return None
        norm = content.lower()
        for m in self._facts:
            if m.content.lower() == norm:
                return None  # dedupe
        fact = MemoryFact.new(content, source_session)
        self._facts.append(fact)
        self._save()
        return fact

    def remove(self, query: str) -> int:
        """Xoá theo id hoặc theo keyword (substring match, case-insensitive).

        Trả về số fact bị xoá.
        """
        if not query:
            return 0
        q_lower = query.lower()
        before = len(self._facts)
        self._facts = [
            m for m in self._facts
            if m.id != query and q_lower not in m.content.lower()
        ]
        removed = before - len(self._facts)
        if removed:
            self._save()
        return removed

    def clear(self) -> int:
        n = len(self._facts)
        self._facts.clear()
        self._save()
        return n

    def format_for_prompt(self) -> str:
        """Render memory list để chèn vào system prompt. Rỗng nếu chưa có fact."""
        if not self._facts:
            return ""
        lines = [f"- {m.content}" for m in self._facts]
        return "Thông tin đã ghi nhớ về người dùng:\n" + "\n".join(lines)
