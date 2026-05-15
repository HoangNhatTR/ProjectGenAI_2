"""Persistent chat sessions cho RAG bot.

Mỗi session = 1 file JSON ở <data_dir>/sessions/<id>.json. Lưu toàn bộ
lịch sử user/assistant để khôi phục ở lần chạy sau.
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


SESSIONS_DIRNAME = "sessions"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _make_id() -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = uuid.uuid4().hex[:4]
    return f"{ts}-{suffix}"


@dataclass
class Session:
    id: str
    name: str
    created_at: str
    updated_at: str
    messages: list[dict] = field(default_factory=list)
    # Rolling summary: messages[0:summary_until] đã được gói gọn vào `summary`.
    # Khi gửi cho LLM ta chỉ cần inject `summary` + messages[summary_until:].
    summary: str = ""
    summary_until: int = 0

    @classmethod
    def new(cls, name: str = "") -> "Session":
        now = _now_iso()
        return cls(
            id=_make_id(),
            name=name.strip() or "Phiên mới",
            created_at=now,
            updated_at=now,
            messages=[],
        )

    @classmethod
    def from_json(cls, data: dict) -> "Session":
        return cls(
            id=data["id"],
            name=data.get("name", "Phiên mới"),
            created_at=data.get("created_at", _now_iso()),
            updated_at=data.get("updated_at", _now_iso()),
            messages=data.get("messages", []),
            summary=data.get("summary", ""),
            summary_until=int(data.get("summary_until", 0)),
        )

    def to_json(self) -> dict:
        return asdict(self)

    def append(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})


class SessionStore:
    """JSON-file store cho session chat."""

    def __init__(self, base_dir: Path | str):
        self.dir = Path(base_dir) / SESSIONS_DIRNAME
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, session_id: str) -> Path:
        return self.dir / f"{session_id}.json"

    def save(self, session: Session) -> None:
        """Ghi atomic: viết tmp xong rename. Tránh file rỗng nếu crash giữa chừng."""
        session.updated_at = _now_iso()
        path = self._path(session.id)
        fd, tmp_path = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(session.to_json(), f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load(self, session_id: str) -> Optional[Session]:
        path = self._path(session_id)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            return Session.from_json(json.load(f))

    def list_recent(self, limit: int = 20) -> list[Session]:
        """Trả các session sắp xếp theo mtime giảm dần (mới nhất trước)."""
        files = sorted(
            self.dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        out: list[Session] = []
        for p in files[:limit]:
            try:
                with p.open(encoding="utf-8") as f:
                    out.append(Session.from_json(json.load(f)))
            except (json.JSONDecodeError, OSError, KeyError):
                continue
        return out

    def delete(self, session_id: str) -> bool:
        path = self._path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def find(self, query: str) -> Optional[Session]:
        """Tìm theo full id, prefix id, hoặc tên (case-insensitive)."""
        if not query:
            return None
        s = self.load(query)
        if s is not None:
            return s
        q_lower = query.lower()
        candidates = self.list_recent(limit=200)
        for s in candidates:
            if s.name.lower() == q_lower:
                return s
        for s in candidates:
            if s.id.startswith(query):
                return s
        return None
