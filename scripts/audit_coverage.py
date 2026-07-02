"""Audit coverage: phát hiện văn bản hiện hành QUAN TRỌNG bị ingest THIẾU.

Bài học NĐ 168 (chỉ 5/57 Điều do docAbs) → quét chunk_count + số Điều phân biệt
của các luật/nghị định trọng yếu. Cờ ⚠ nếu số Điều thấp bất thường so với kỳ vọng.

Read-only. python -m scripts.audit_coverage
"""
from __future__ import annotations

import sys

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from opensearchpy import OpenSearch
from src import config

cli = OpenSearch(hosts=[config.OPENSEARCH_URL], timeout=120)
IDX = config.OPENSEARCH_INDEX

# (doc_number, nhãn, số Điều thực tế kỳ vọng ~). exp=0 nếu không chắc (chỉ xem).
# Nguồn số Điều: bản luật hiện hành. Cờ ⚠ khi articles_in_corpus < 60% kỳ vọng.
KEY_DOCS = [
    # ── Bộ luật / Luật gốc (hàng trăm Điều) ──
    ("100/2015/QH13", "Bộ luật Hình sự 2015", 426),
    ("91/2015/QH13",  "Bộ luật Dân sự 2015", 689),
    ("45/2019/QH14",  "Bộ luật Lao động 2019", 220),
    ("101/2015/QH13", "Bộ luật Tố tụng Hình sự 2015", 510),
    ("92/2015/QH13",  "Bộ luật Tố tụng Dân sự 2015", 517),
    ("31/2024/QH15",  "Luật Đất đai 2024", 260),
    ("52/2014/QH13",  "Luật Hôn nhân & Gia đình 2014", 133),
    ("59/2020/QH14",  "Luật Doanh nghiệp 2020", 218),
    ("61/2020/QH14",  "Luật Đầu tư 2020", 77),
    ("15/2012/QH13",  "Luật Xử lý VPHC 2012", 142),
    ("36/2024/QH15",  "Luật Trật tự ATGT đường bộ 2024", 89),
    ("35/2024/QH15",  "Luật Đường bộ 2024", 86),
    ("27/2023/QH15",  "Luật Nhà ở 2023", 198),
    ("29/2023/QH15",  "Luật Kinh doanh BĐS 2023", 83),
    ("24/2008/QH12",  "Luật Quốc tịch (cũ) / tham chiếu", 0),
    ("43/2013/QH13",  "Luật Đấu thầu (cũ)", 0),
    ("22/2023/QH15",  "Luật Khám bệnh, chữa bệnh 2023", 121),
    # ── Nghị định then chốt hiện hành ──
    ("168/2024/NĐ-CP", "NĐ 168/2024 xử phạt GT (đã fix)", 57),
    ("145/2020/NĐ-CP", "NĐ 145/2020 hướng dẫn Bộ luật LĐ", 115),
    ("01/2021/NĐ-CP",  "NĐ 01/2021 đăng ký doanh nghiệp", 0),
    ("96/2023/NĐ-CP",  "NĐ 96/2023 hướng dẫn Luật Khám chữa bệnh", 0),
    ("88/2024/NĐ-CP",  "NĐ 88/2024 bồi thường khi thu hồi đất", 0),
    ("102/2024/NĐ-CP", "NĐ 102/2024 hướng dẫn Luật Đất đai", 0),
    ("125/2020/NĐ-CP", "NĐ 125/2020 xử phạt thuế, hóa đơn", 0),
    ("144/2021/NĐ-CP", "NĐ 144/2021 xử phạt ANTT, an toàn XH", 0),
    ("100/2019/NĐ-CP", "NĐ 100/2019 GT (đã hết hiệu lực 1 phần)", 0),
]


def audit(dn: str):
    cnt = cli.count(index=IDX, body={"query": {"term": {"doc_number": dn}}})["count"]
    if cnt == 0:
        return cnt, 0
    r = cli.search(index=IDX, body={
        "size": 0, "query": {"term": {"doc_number": dn}},
        "aggs": {"a": {"cardinality": {"field": "article"}}},
    })
    return cnt, int(r["aggregations"]["a"]["value"])


print(f"{'doc_number':<16}{'chunk':>8}{'Điều':>7}{'kỳ vọng':>9}  văn bản / cờ")
print("-" * 86)
flags = []
for dn, label, exp in KEY_DOCS:
    cnt, arts = audit(dn)
    flag = ""
    if cnt == 0:
        flag = "❌ THIẾU HẲN"
        flags.append((dn, label, flag))
    elif exp and arts < exp * 0.6:
        flag = f"⚠ THIẾU (chỉ {arts}/{exp} Điều ≈ {arts*100//exp}%)"
        flags.append((dn, label, flag))
    exp_s = str(exp) if exp else "?"
    print(f"{dn:<16}{cnt:>8}{arts:>7}{exp_s:>9}  {label[:34]:<34} {flag}")

print("\n" + "=" * 86)
if flags:
    print(f"⚠ {len(flags)} VĂN BẢN CẦN XỬ LÝ:")
    for dn, label, flag in flags:
        print(f"   {dn:<16} {label[:40]:<40} {flag}")
else:
    print("✅ Không phát hiện văn bản trọng yếu nào thiếu rõ rệt.")
