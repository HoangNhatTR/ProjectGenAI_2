"""Conversation State Management cho Legal AI Agent.

Tracks ngữ cảnh pháp lý xuyên suốt hội thoại: chủ đề, hành vi vi phạm,
loại phương tiện, văn bản luật gần nhất, entities đã nhắc đến.

Giúp Router viết lại câu hỏi mơ hồ (vd: "thế ô tô thì sao?" →
"Mức phạt vượt đèn đỏ đối với ô tô là bao nhiêu?") và Generator
cá nhân hóa câu trả lời theo ngữ cảnh.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Optional


# ── Từ điển loại phương tiện ──────────────────────────────────────────────────
VEHICLE_KEYWORDS: dict[str, list[str]] = {
    "xe máy": ["xe máy", "môtô", "mô tô", "hai bánh", "xe gắn máy", "xe moto"],
    "ô tô":   ["ô tô", "xe hơi", "xe con", "xe tải", "xe khách", "xe bus", "xe buýt", "ôtô"],
    "xe đạp": ["xe đạp"],
    "xe điện": ["xe điện", "xe máy điện", "xe đạp điện"],
    "tàu":    ["tàu", "thuyền", "phà", "tàu thủy"],
}

# Patterns nhận diện tham chiếu điều/khoản/văn bản
_LAW_REF_PATTERNS = [
    r'\d+/\d{4}/[A-ZĐÂĂÊÔƠƯ\-]+',        # 100/2019/NĐ-CP
    r'Nghị\s+định\s+(?:số\s+)?\d+',       # Nghị định 100
    r'Thông\s+tư\s+(?:số\s+)?\d+',        # Thông tư 24
    r'Luật\s+[A-ZĐÂĂÊÔƠƯa-zđâăêôơư ]+\d{4}',  # Luật Giao thông 2008
    r'Điều\s+\d+',                          # Điều 6
    r'Khoản\s+\d+',                         # Khoản 2
]

# Keywords nhận diện chủ đề pháp lý
_TOPIC_KEYWORDS: dict[str, list[str]] = {
    "vi phạm giao thông": [
        "vượt đèn đỏ", "quá tốc độ", "nồng độ cồn", "ma túy", "không đội mũ",
        "phạt nguội", "thu hồi bằng", "tước quyền", "giao thông", "lái xe",
        "GPLX", "giấy phép lái xe",
    ],
    "hợp đồng": ["hợp đồng", "vi phạm hợp đồng", "bồi thường", "đặt cọc"],
    "đất đai": ["đất đai", "nhà ở", "tranh chấp đất", "quyền sử dụng đất", "sổ đỏ"],
    "lao động": ["sa thải", "lao động", "hợp đồng lao động", "bảo hiểm xã hội", "tiền lương"],
    "hình sự": ["tội phạm", "truy tố", "kết án", "tù giam", "phạt tiền hình sự"],
    "doanh nghiệp": ["công ty", "doanh nghiệp", "cổ phần", "thành lập", "giải thể"],
}


def _detect_vehicle(text: str) -> Optional[str]:
    text_lower = text.lower()
    for vehicle, keywords in VEHICLE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return vehicle
    return None


def _detect_topic(text: str) -> Optional[str]:
    text_lower = text.lower()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return topic
    return None


def _extract_law_refs(text: str) -> list[str]:
    """Trích xuất tham chiếu văn bản/điều khoản từ text."""
    refs: list[str] = []
    for pattern in _LAW_REF_PATTERNS:
        refs.extend(re.findall(pattern, text, re.IGNORECASE))
    # Dedupe, preserve order
    seen: set[str] = set()
    result: list[str] = []
    for r in refs:
        r_norm = r.strip()
        if r_norm and r_norm not in seen:
            seen.add(r_norm)
            result.append(r_norm)
    return result


@dataclass
class ConversationState:
    """Trạng thái ngữ cảnh pháp lý của hội thoại.

    Được cập nhật sau mỗi lượt (từ câu hỏi + câu trả lời) và inject
    vào Router (viết lại query) và Generator (cá nhân hóa câu trả lời).
    """

    current_legal_topic: Optional[str] = None    # "vi phạm giao thông"
    current_act: Optional[str] = None            # "vượt đèn đỏ"
    vehicle_type: Optional[str] = None           # "xe máy"
    last_retrieved_law: Optional[str] = None     # "Nghị định 100/2019/NĐ-CP"
    law_references: list[str] = field(default_factory=list)  # các tham chiếu đã thấy
    entities: list[str] = field(default_factory=list)        # từ khóa quan trọng
    pending_clarification: Optional[str] = None  # câu hỏi còn lơ lửng

    # ── Cập nhật state ────────────────────────────────────────────────────────

    def update_from_question(self, question: str) -> None:
        """Cập nhật state từ câu hỏi mới của user."""
        vehicle = _detect_vehicle(question)
        if vehicle:
            self.vehicle_type = vehicle

        topic = _detect_topic(question)
        if topic and not self.current_legal_topic:
            self.current_legal_topic = topic

        for ref in _extract_law_refs(question):
            if ref not in self.law_references:
                self.law_references.append(ref)

    def update_from_answer(self, answer: str, retrieved_sources: list[str]) -> None:
        """Cập nhật state từ câu trả lời và nguồn đã retrieve."""
        for ref in _extract_law_refs(answer):
            if ref not in self.law_references:
                self.law_references.append(ref)

        topic = _detect_topic(answer)
        if topic and not self.current_legal_topic:
            self.current_legal_topic = topic

        if retrieved_sources:
            self.last_retrieved_law = retrieved_sources[0]

    def set_act(self, act: str) -> None:
        """Đặt hành vi vi phạm chính (thường từ router/planner extract)."""
        if act and act.strip():
            self.current_act = act.strip()

    def reset_turn(self) -> None:
        """Xóa pending_clarification sau mỗi lượt giải quyết xong."""
        self.pending_clarification = None

    # ── Format để inject vào prompt ───────────────────────────────────────────

    def to_context_string(self) -> str:
        """Render state thành đoạn text để chèn vào system prompt / router prompt."""
        parts: list[str] = []
        if self.current_legal_topic:
            parts.append(f"Chủ đề pháp lý: {self.current_legal_topic}")
        if self.current_act:
            parts.append(f"Hành vi / tình huống: {self.current_act}")
        if self.vehicle_type:
            parts.append(f"Loại phương tiện: {self.vehicle_type}")
        if self.last_retrieved_law:
            parts.append(f"Văn bản gần nhất đã tra: {self.last_retrieved_law}")
        if self.law_references:
            refs_str = ", ".join(self.law_references[-5:])  # chỉ hiện 5 gần nhất
            parts.append(f"Tham chiếu đã đề cập: {refs_str}")
        if not parts:
            return ""
        return "Ngữ cảnh hội thoại hiện tại:\n" + "\n".join(f"  - {p}" for p in parts)

    def has_context(self) -> bool:
        return any([
            self.current_legal_topic,
            self.current_act,
            self.vehicle_type,
            self.last_retrieved_law,
        ])

    # ── Persistence ───────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ConversationState":
        valid_fields = cls.__dataclass_fields__
        return cls(**{k: v for k, v in data.items() if k in valid_fields})

    def clear(self) -> None:
        """Reset toàn bộ state (dùng khi /clear session)."""
        self.current_legal_topic = None
        self.current_act = None
        self.vehicle_type = None
        self.last_retrieved_law = None
        self.law_references.clear()
        self.entities.clear()
        self.pending_clarification = None
