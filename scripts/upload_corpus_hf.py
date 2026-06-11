"""Nén corpus sạch data/raw → upload lên HuggingFace dataset (private).

Colab sẽ tải archive này về thay vì re-crawl từ HF gốc — đảm bảo embed
ĐÚNG bộ corpus đã dọn (loại VB địa phương + khử trùng lặp), bao gồm cả
nhóm all_laws/an_le từ vbpl.vn vốn không có trên HF dataset gốc.

Chạy:  python -m scripts.upload_corpus_hf
Cần:   HF_TOKEN trong .env (quyền write)
"""
from __future__ import annotations

import os
import sys
import tarfile
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# .env set TRANSFORMERS_OFFLINE=1 cho inference local — phải tắt offline mode
# TRƯỚC khi import huggingface_hub, nếu không API call sẽ bị chặn
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

RAW_DIR = ROOT / "data" / "raw"
ARCHIVE = ROOT / "data" / "exports" / "corpus_clean.tar.gz"
HF_REPO = "HoangNhat1304/legalai-corpus-clean"


def main() -> None:
    token = os.getenv("HF_TOKEN", "")
    if not token:
        sys.exit("Thiếu HF_TOKEN trong .env")

    n_files = sum(1 for _ in RAW_DIR.rglob("*.txt"))
    if ARCHIVE.exists() and ARCHIVE.stat().st_size > 1e6 and "--force" not in sys.argv:
        mb = ARCHIVE.stat().st_size / 1e6
        print(f"[1/3] Archive đã có ({mb:,.0f} MB) — tái dùng (xóa hoặc --force để nén lại)")
    else:
        print(f"[1/3] Nén {n_files:,} file từ {RAW_DIR} ...")
        ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with tarfile.open(ARCHIVE, "w:gz", compresslevel=6) as tar:
            # arcname='raw' → giải nén vào data/ sẽ tạo data/raw/
            tar.add(RAW_DIR, arcname="raw")
        mb = ARCHIVE.stat().st_size / 1e6
        print(f"  ✓ {ARCHIVE.name}: {mb:,.0f} MB ({(time.time()-t0)/60:.1f} phút)")

    print(f"[2/3] Tạo repo dataset {HF_REPO} (private)...")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(HF_REPO, repo_type="dataset", private=True, exist_ok=True)

    print(f"[3/3] Upload {mb:,.0f} MB lên HF...")
    t0 = time.time()
    api.upload_file(
        path_or_fileobj=str(ARCHIVE),
        repo_id=HF_REPO,
        repo_type="dataset",
        path_in_repo="corpus_clean.tar.gz",
        commit_message=f"Corpus sạch {n_files:,} VB (loại địa phương + khử trùng lặp)",
    )
    print(f"  ✓ Upload xong ({(time.time()-t0)/60:.1f} phút)")
    print(f"\nColab tải về bằng:")
    print(f"  hf_hub_download(repo_id='{HF_REPO}', repo_type='dataset',")
    print(f"                  filename='corpus_clean.tar.gz', token=HF_TOKEN)")


if __name__ == "__main__":
    main()
