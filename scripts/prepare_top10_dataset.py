"""Tách top 10 luật quan trọng ra thành dataset riêng cho RAG vs Graph-RAG eval.

Output:
    data/comparison/top10_laws/<doc_number>.txt   # copy nguyên file raw
    data/comparison/top10_laws/manifest.json      # metadata + stats từng luật

Chạy:
    python -m scripts.prepare_top10_dataset
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.chunking import _iter_articles
from src.kg.structural_extractor import _normalize_for_kg
from src.parsing import load_document
from scripts.ingest import iter_raw_files


# Top 15 luật — match với DEFAULT_TOP_LAWS trong build_semantic_kg.py
# Chọn theo tiêu chí: (1) còn hiệu lực hoặc mới nhất, (2) đời sống thường ngày, (3) đa lĩnh vực
TOP10 = [
    # === Bộ luật & Luật cốt lõi ===
    ("91/2015/QH13",  "Bộ luật Dân sự",                "dan_su"),
    ("100/2015/QH13", "Bộ luật Hình sự",               "hinh_su"),
    ("101/2015/QH13", "Bộ luật Tố tụng hình sự",       "tths"),
    ("45/2019/QH14",  "Bộ luật Lao động",              "lao_dong"),
    ("31/2024/QH15",  "Luật Đất đai",                  "dat_dai"),
    ("59/2020/QH14",  "Luật Doanh nghiệp",             "doanh_nghiep"),
    ("36/2024/QH15",  "Luật Trật tự ATGT đường bộ",    "giao_thong"),
    ("52/2014/QH13",  "Luật Hôn nhân và Gia đình",     "hon_nhan_gd"),
    ("108/2025/QH15", "Luật Quản lý thuế",             "quan_ly_thue"),
    ("143/2025/QH15", "Luật Đầu tư",                   "dau_tu"),
    # === 5 luật phổ biến đời thường thêm (top 15) ===
    ("41/2024/QH15",  "Luật Bảo hiểm xã hội",          "bhxh"),
    ("27/2023/QH15",  "Luật Nhà ở",                    "nha_o"),
    ("15/2023/QH15",  "Luật Khám bệnh, chữa bệnh",     "kham_chua_benh"),
    ("73/2021/QH14",  "Luật Phòng chống ma túy",       "ma_tuy"),
    ("19/2023/QH15",  "Luật Bảo vệ quyền lợi NTD",     "bvntd"),
]


def main() -> None:
    out_dir = config.DATA_DIR / "comparison" / "top10_laws"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}\n")

    manifest = {
        "description": "Top 10 luật quan trọng — dataset cho RAG vs Graph-RAG comparison",
        "total_laws": len(TOP10),
        "laws": [],
    }

    total_articles = 0
    total_chars = 0
    missing = []

    for i, (doc_num, ten_luat, slug) in enumerate(TOP10, 1):
        print(f"[{i:2d}] {doc_num}: {ten_luat}")

        # Tìm file raw
        found = None
        for path, meta in iter_raw_files(config.RAW_DIR):
            if meta.doc_number == doc_num:
                found = (path, meta)
                break

        if not found:
            print(f"     ⚠ Không tìm thấy file raw")
            missing.append(doc_num)
            continue

        src_path, meta = found

        # Copy file
        dest_name = f"{slug}_{doc_num.replace('/', '_')}.txt"
        dest_path = out_dir / dest_name
        shutil.copy2(src_path, dest_path)

        # Đếm điều
        try:
            doc = load_document(src_path, meta)
            normalized = _normalize_for_kg(doc.text)
            n_articles = sum(
                1 for c, a, _ in _iter_articles(normalized) if a is not None
            )
        except Exception as exc:
            print(f"     ⚠ Lỗi parse: {exc}")
            n_articles = 0

        char_count = src_path.stat().st_size
        total_articles += n_articles
        total_chars += char_count

        manifest["laws"].append({
            "doc_number": doc_num,
            "title": meta.title or ten_luat,
            "slug": slug,
            "source_url": meta.source,
            "doc_type": meta.doc_type,
            "issued_date": meta.issued_date,
            "status": meta.status,
            "original_file": str(src_path.relative_to(config.RAW_DIR.parent)),
            "copied_file": str(dest_path.relative_to(config.DATA_DIR.parent)),
            "n_articles": n_articles,
            "size_bytes": char_count,
        })

        print(f"     → {dest_name} ({n_articles} điều, {char_count/1024:.1f} KB)")

    manifest["totals"] = {
        "n_laws_copied": len(manifest["laws"]),
        "n_articles": total_articles,
        "total_size_bytes": total_chars,
        "total_size_mb": round(total_chars / (1024 * 1024), 2),
    }
    if missing:
        manifest["missing"] = missing

    # Lưu manifest
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"Đã copy {len(manifest['laws'])}/{len(TOP10)} luật")
    print(f"Tổng: {total_articles:,} điều, {total_chars/(1024*1024):.2f} MB")
    if missing:
        print(f"⚠ Thiếu: {missing}")
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
