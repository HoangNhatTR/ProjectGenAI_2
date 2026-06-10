"""Post-crawl pipeline: ingest → build_bm25 → enrich_metadata → analyze_data.

Chạy sau khi crawl xong để nạp toàn bộ dữ liệu mới vào hệ thống.

Cách chạy:
    python -m scripts.post_crawl              # full pipeline
    python -m scripts.post_crawl --skip-ingest  # bỏ qua ingest (nếu đã chạy)
    python -m scripts.post_crawl --reset        # reset vectorstore trước khi ingest
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], step: str) -> bool:
    t0 = time.time()
    print(f"\n{'='*60}")
    print(f"  [{step}]")
    print(f"  Lệnh: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=ROOT)
    elapsed = round(time.time() - t0, 1)
    ok = result.returncode == 0
    status = "✓ OK" if ok else f"✗ LỖI (code={result.returncode})"
    print(f"\n  {status} — {elapsed}s")
    return ok


def main() -> None:
    p = argparse.ArgumentParser(description="Post-crawl pipeline")
    p.add_argument("--skip-ingest",  action="store_true")
    p.add_argument("--skip-enrich",  action="store_true")
    p.add_argument("--skip-bm25",    action="store_true")
    p.add_argument("--skip-analyze", action="store_true")
    p.add_argument("--reset",        action="store_true",
                   help="Reset vectorstore trước khi ingest")
    args = p.parse_args()

    print(f"\nPost-crawl pipeline — {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

    steps = []

    # ── 1. Enrich metadata (thêm LINH_VUC, quan hệ VB) ───────────────────────
    if not args.skip_enrich:
        steps.append(("Enrich metadata", [
            sys.executable, "-m", "scripts.enrich_metadata"
        ]))
        # Enrich riêng cho hf_laws nếu có
        hf_dir = ROOT / "data" / "raw" / "hf_laws"
        if hf_dir.exists():
            steps.append(("Enrich metadata (hf_laws)", [
                sys.executable, "-m", "scripts.enrich_metadata", "--dir", "hf_laws"
            ]))

    # ── 2. Ingest vào vectorstore ─────────────────────────────────────────────
    if not args.skip_ingest:
        ingest_cmd = [sys.executable, "-m", "scripts.ingest", "--skip-existing"]
        if args.reset:
            ingest_cmd = [sys.executable, "-m", "scripts.ingest", "--reset"]
        steps.append(("Ingest → vectorstore", ingest_cmd))

    # ── 3. Build BM25 index ───────────────────────────────────────────────────
    if not args.skip_bm25:
        steps.append(("Build BM25 index", [
            sys.executable, "-m", "scripts.build_bm25"
        ]))

    # ── 4. Analyze & báo cáo ─────────────────────────────────────────────────
    if not args.skip_analyze:
        steps.append(("Analyze data", [
            sys.executable, "-m", "scripts.analyze_data",
            "--export", "data/data_report.json"
        ]))

    # ── Run all steps ─────────────────────────────────────────────────────────
    results = []
    for step_name, cmd in steps:
        ok = run(cmd, step_name)
        results.append((step_name, ok))
        if not ok:
            print(f"\n  ⚠ Bước '{step_name}' thất bại. Tiếp tục...")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  TỔNG KẾT POST-CRAWL PIPELINE")
    print(f"{'='*60}")
    for name, ok in results:
        icon = "✓" if ok else "✗"
        print(f"  {icon} {name}")

    all_ok = all(ok for _, ok in results)
    if all_ok:
        print("\n  Pipeline hoàn thành! Hệ thống đã cập nhật dữ liệu mới.")
        print("  Khởi động lại app: python app.py")
    else:
        print("\n  Một số bước có lỗi. Kiểm tra output bên trên.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
