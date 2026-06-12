"""Export toàn bộ chunks + embeddings từ Chroma → parquet shards (portable).

Vectors trong Chroma nằm ở format hnswlib (data_level0.bin) — không DB nào
import trực tiếp được. Script này xuất ra parquet (id, text, metadata JSON,
embedding float32[1024]) làm format trao đổi: OpenSearch/Qdrant/Milvus/
pgvector đều ingest được từ đây, không bao giờ phải re-embed.

CHẠY TRÊN COLAB HIGH-RAM (chroma .get(ids) cần load HNSW ~21GB vào RAM):
    %cd /content/ProjectGenAI_2
    !python -m scripts.export_embeddings --upload

Output: data/exports/embeddings/shard_NNNN.parquet (~100k rows/shard)
--upload: đẩy từng shard lên HF dataset rồi xóa local (tiết kiệm disk).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

os.environ.setdefault("HF_HUB_OFFLINE", "0")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

from src import config  # noqa: E402

HF_REPO = "HoangNhat1304/legalai-vectorstore"
SHARD_ROWS = 100_000
GET_BATCH = 500


def _all_ids(db_path: Path) -> list[str]:
    """Toàn bộ embedding_id qua SQLite (nhanh, không qua API phân trang)."""
    import sqlite3
    con = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        return [r[0] for r in con.execute("SELECT embedding_id FROM embeddings")]
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--upload", action="store_true", help="Upload từng shard lên HF rồi xóa local")
    p.add_argument("--out", default=str(config.DATA_DIR / "exports" / "embeddings"))
    args = p.parse_args()

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import chromadb
    from chromadb.config import Settings

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = None
    if args.upload:
        from huggingface_hub import HfApi
        token = os.getenv("HF_TOKEN", "")
        assert token, "Thiếu HF_TOKEN (env hoặc .env)"
        api = HfApi(token=token)

    client = chromadb.PersistentClient(
        path=str(config.VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    coll = client.get_collection(config.COLLECTION_NAME)

    ids = _all_ids(Path(config.VECTORSTORE_DIR) / "chroma.sqlite3")
    total = len(ids)
    print(f"Export {total:,} chunks → shards {SHARD_ROWS:,} rows "
          f"({(total + SHARD_ROWS - 1)//SHARD_ROWS} shards)")

    schema = pa.schema([
        ("id", pa.string()),
        ("text", pa.string()),
        ("metadata", pa.string()),                       # JSON string
        ("embedding", pa.list_(pa.float32(), EMBED := 1024)),
    ])

    buf_ids: list[str] = []
    buf_txt: list[str] = []
    buf_meta: list[str] = []
    buf_emb: list[np.ndarray] = []
    shard_no = 0
    t0 = time.time()

    def _flush_shard() -> None:
        nonlocal shard_no, buf_ids, buf_txt, buf_meta, buf_emb
        if not buf_ids:
            return
        table = pa.Table.from_arrays(
            [
                pa.array(buf_ids, pa.string()),
                pa.array(buf_txt, pa.string()),
                pa.array(buf_meta, pa.string()),
                pa.FixedSizeListArray.from_arrays(
                    pa.array(np.concatenate(buf_emb).astype("float32")), EMBED
                ).cast(pa.list_(pa.float32(), EMBED)),
            ],
            schema=schema,
        )
        path = out_dir / f"shard_{shard_no:04d}.parquet"
        pq.write_table(table, path, compression="zstd")
        mb = path.stat().st_size / 1e6
        done = shard_no * SHARD_ROWS + len(buf_ids)
        rate = done / (time.time() - t0)
        print(f"  shard_{shard_no:04d}: {len(buf_ids):,} rows, {mb:.0f} MB "
              f"| tổng {done:,}/{total:,} ({rate:.0f} rows/s)")
        if api is not None:
            api.upload_file(
                path_or_fileobj=str(path), repo_id=HF_REPO, repo_type="dataset",
                path_in_repo=f"embeddings_export/{path.name}",
            )
            path.unlink()
        shard_no += 1
        buf_ids, buf_txt, buf_meta, buf_emb = [], [], [], []

    for i in range(0, total, GET_BATCH):
        batch_ids = ids[i:i + GET_BATCH]
        res = coll.get(ids=batch_ids, include=["embeddings", "documents", "metadatas"])
        for cid, doc, meta, emb in zip(
            res["ids"], res["documents"], res["metadatas"], res["embeddings"]
        ):
            buf_ids.append(cid)
            buf_txt.append(doc or "")
            buf_meta.append(json.dumps(meta or {}, ensure_ascii=False))
            buf_emb.append(np.asarray(emb, dtype="float32"))
        if len(buf_ids) >= SHARD_ROWS:
            _flush_shard()

    _flush_shard()
    print(f"\n✓ Export xong {total:,} chunks sau {(time.time()-t0)/60:.1f} phút")
    if api is not None:
        print(f"  → HF: {HF_REPO}/embeddings_export/")


if __name__ == "__main__":
    main()
