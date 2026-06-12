# OpenSearch backend — Runbook

Chuyển hệ thống từ Chroma local sang **OpenSearch server**: vector kNN (HNSW
lucene, page-cache friendly) + BM25 native trong **một index** — backend query
trực tiếp qua API, không cần tải 45GB Chroma về mỗi máy.

## Yêu cầu máy chạy OpenSearch

- RAM ≥ 16GB (heap 4GB + page cache cho index ~25GB)
- Disk ≥ 60GB trống (index + bản tải parquet tạm)
- Docker Desktop (Windows/Mac) hoặc Docker Engine (Linux)

## Bước 1 — Export embeddings từ Chroma → parquet (chạy 1 lần trên Colab)

Vectors trong Chroma là format hnswlib, không import trực tiếp được — export
ra parquet làm format trao đổi chuẩn (dùng được cho cả Qdrant/Milvus/pgvector
sau này, không bao giờ phải re-embed).

Trên **Colab High-RAM** (cần ~25GB RAM để Chroma đọc vectors):

```python
# Cell: tải store + export
%cd /content
!git clone https://github.com/HoangNhatTR/ProjectGenAI_2.git
%cd ProjectGenAI_2
!pip install -q -r requirements.txt

from google.colab import userdata
import os
os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")

from huggingface_hub import snapshot_download
snapshot_download(repo_id="HoangNhat1304/legalai-vectorstore", repo_type="dataset",
                  local_dir="/content/store", allow_patterns=["chroma/**"],
                  token=os.environ["HF_TOKEN"])

os.environ["VECTORSTORE_DIR"] = "/content/store/chroma"
!python -m scripts.export_embeddings --upload
```

Kết quả: `embeddings_export/shard_*.parquet` (~50 shards, tổng ~22GB) trên
HF dataset `HoangNhat1304/legalai-vectorstore`.

## Bước 2 — Dựng OpenSearch trên server

```bash
docker compose -f docker-compose.opensearch.yml up -d
curl http://localhost:9200        # → JSON version là OK
```

⚠ Security plugin đang TẮT và port chỉ bind 127.0.0.1 — nếu cần truy cập từ
máy khác trong LAN, đổi `127.0.0.1:9200:9200` thành `9200:9200` và cân nhắc
bật security/đặt firewall.

## Bước 3 — Ingest parquet vào OpenSearch (chạy trên server)

```bash
pip install opensearch-py pyarrow huggingface_hub
# HF_TOKEN trong .env
python -m scripts.ingest_opensearch --reset
```

Script tự tải từng shard từ HF → bulk index → xóa shard (đỡ tốn disk).
Ước tính: ~30–60 phút cho 4.9M docs (nghẽn ở HNSW build phía OpenSearch).

## Bước 4 — Trỏ backend sang OpenSearch

Trong `.env`:

```
VECTOR_BACKEND=opensearch
OPENSEARCH_URL=http://localhost:9200
OPENSEARCH_INDEX=legal_chunks
```

Chạy `uvicorn api:app --port 8000` — log khởi động sẽ in:
`[API] Backend      : OpenSearch @ http://localhost:9200 (vector + BM25)`

Vẫn cần local: model embedding BGE-M3 (encode câu hỏi, ~2.3GB download lần
đầu, chạy CPU được) + `parent_store.db` (tải từ HF về, trỏ `PARENT_STORE_PATH`).

## Kiến trúc sau khi chuyển

```
Máy local / backend       Server OpenSearch (RAM 16GB+)
┌─────────────────┐        ┌──────────────────────────┐
│ api.py / UI     │ HTTP   │ index legal_chunks       │
│ BGE-M3 (query)  ├───────►│  - BM25 (text)           │
│ Reranker CE     │        │  - kNN HNSW (embedding)  │
│ parent_store.db │        │  - metadata filters      │
└─────────────────┘        └──────────────────────────┘
```

Rollback: đặt lại `VECTOR_BACKEND=chroma` — không đụng gì khác.
