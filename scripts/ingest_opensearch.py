"""Ingest parquet shards (từ export_embeddings.py) vào OpenSearch.

Chạy TRÊN SERVER (máy chạy OpenSearch) — stream từng shard, RAM thấp:
    # OpenSearch đã chạy (docker compose -f docker-compose.opensearch.yml up -d)
    python -m scripts.ingest_opensearch                 # tải shards từ HF, ingest, xóa
    python -m scripts.ingest_opensearch --local-dir D:\\shards   # đã tải sẵn
    python -m scripts.ingest_opensearch --reset         # xóa index cũ trước

Tối ưu trong lúc ingest: tắt refresh + replicas, bulk 500 docs/request.
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
# Trình tải Rust đa kết nối — nhanh hơn nhiều lần trên đường truyền
# quốc tế bị bóp băng thông (cần: pip install hf_transfer)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from src import config  # noqa: E402
from src.opensearch_store import OpenSearchVectorStore  # noqa: E402

HF_REPO = "HoangNhat1304/legalai-vectorstore"
HF_PREFIX = "embeddings_export/"


def _iter_shard_paths(local_dir: str | None):
    """Yield (tên shard, path local, callback dọn dẹp)."""
    if local_dir:
        for p in sorted(Path(local_dir).glob("shard_*.parquet")):
            yield p.name, p, (lambda: None)
        return

    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi, hf_hub_download
    token = os.getenv("HF_TOKEN", "")
    assert token, "Thiếu HF_TOKEN trong .env để tải shards từ HF"
    api = HfApi(token=token)
    names = sorted(
        f for f in api.list_repo_files(HF_REPO, repo_type="dataset")
        if f.startswith(HF_PREFIX) and f.endswith(".parquet")
    )
    assert names, f"Không thấy shard nào tại {HF_REPO}/{HF_PREFIX} — chạy export_embeddings trước"
    print(f"Tìm thấy {len(names)} shards trên HF", flush=True)

    def _dl(name: str) -> Path:
        return Path(hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=name,
            token=token, local_dir="data/_os_ingest_tmp",
        ))

    # Prefetch: tải shard i+1 trong lúc shard i đang được index —
    # download (nghẽn mạng) và indexing (nghẽn CPU/disk) chồng lên nhau
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(_dl, names[0])
        for i, name in enumerate(names):
            path = fut.result()
            if i + 1 < len(names):
                fut = pool.submit(_dl, names[i + 1])
            yield name, path, path.unlink


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--local-dir", default=None, help="Thư mục chứa shard_*.parquet đã tải sẵn")
    p.add_argument("--reset", action="store_true", help="Xóa index cũ trước khi ingest")
    args = p.parse_args()

    import pyarrow.parquet as pq
    from opensearchpy import helpers

    store = OpenSearchVectorStore(config.OPENSEARCH_URL, config.OPENSEARCH_INDEX)
    cli = store._connect()
    info = cli.info()
    print(f"OpenSearch {info['version']['number']} @ {config.OPENSEARCH_URL}")

    if args.reset:
        print("Reset index...")
        store.reset()
    else:
        store.ensure_index()

    # Tăng tốc bulk: tắt refresh trong lúc ingest
    cli.indices.put_settings(index=config.OPENSEARCH_INDEX,
                             body={"index": {"refresh_interval": "-1"}})

    total = 0
    t0 = time.time()
    try:
        for name, path, cleanup in _iter_shard_paths(args.local_dir):
            table = pq.read_table(path)
            ids = table.column("id").to_pylist()
            texts = table.column("text").to_pylist()
            metas = table.column("metadata").to_pylist()
            # Decode embeddings qua numpy (nhanh hơn nhiều lần to_pylist cả cột)
            emb_np = (
                table.column("embedding").combine_chunks()
                .flatten().to_numpy(zero_copy_only=False)
                .reshape(len(table), -1)
            )

            def _actions():
                for i, (cid, text, meta_json) in enumerate(zip(ids, texts, metas)):
                    doc = json.loads(meta_json) if meta_json else {}
                    doc.pop("chroma:document", None)
                    doc["text"] = text
                    doc["embedding"] = emb_np[i].tolist()
                    yield {"_op_type": "index", "_index": config.OPENSEARCH_INDEX,
                           "_id": cid, "_source": doc}

            # parallel_bulk: serialize + gửi song song 4 luồng — nút thắt
            # client-side 1 luồng từng làm ingest chậm ~5x
            ok = n_err = 0
            for success, _item in helpers.parallel_bulk(
                cli, _actions(), thread_count=4, chunk_size=500,
                queue_size=8, request_timeout=180, raise_on_error=False,
            ):
                if success:
                    ok += 1
                else:
                    n_err += 1
            total += ok
            rate = total / (time.time() - t0)
            print(f"  {name}: +{ok:,} docs (lỗi: {n_err}) | tổng {total:,} | {rate:.0f} docs/s",
                  flush=True)
            cleanup()
    finally:
        cli.indices.put_settings(index=config.OPENSEARCH_INDEX,
                                 body={"index": {"refresh_interval": "30s"}})

    cli.indices.refresh(index=config.OPENSEARCH_INDEX)
    print(f"\n✓ Ingest xong: {store.count():,} docs sau {(time.time()-t0)/60:.1f} phút")

    # Force merge: gộp segments + build HNSW graph cho segment lớn
    # (đi cùng knn.advanced.approximate_threshold) → query nhanh hơn nhiều.
    # Có thể mất 30-60 phút với index ~25GB — chạy ngầm phía server.
    print("Force merge segments (nền, có thể 30-60 phút)...", flush=True)
    try:
        cli.indices.forcemerge(
            index=config.OPENSEARCH_INDEX, max_num_segments=5,
            params={"wait_for_completion": "false"},
        )
        print("  → đã kích hoạt merge nền — theo dõi: GET _cat/segments/legal_chunks")
    except Exception as exc:
        print(f"  [!] forcemerge lỗi (không nghiêm trọng): {exc}")

    print("Đặt trong .env:  VECTOR_BACKEND=opensearch")


if __name__ == "__main__":
    main()
