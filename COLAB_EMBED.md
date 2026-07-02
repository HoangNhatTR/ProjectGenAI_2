# Re-ingest qua Colab GPU (embed nhanh) → ghép vào OpenSearch local

Khâu chậm duy nhất là **embed BGE-M3**. Chạy nó trên Colab GPU (nhanh ~20-50×),
fetch + chunk + ghi index vẫn làm ở **máy local** (OpenSearch chạy local).

```
[LOCAL]  export_to_embed.py   →  to_embed.jsonl (+ parents.jsonl, manifest.json)
   ↑ fetch vbpl + chunk (CPU, ~15')          │ upload
[COLAB]  cell dưới đây (GPU)  →  embedded.parquet (thêm vector)
   │ download
[LOCAL]  merge_embedded.py    →  xoá chunk cũ + add chunk mới + parent_store
```

⚠️ **Bắt buộc cùng model** `BAAI/bge-m3` + `normalize_embeddings=True` thì vector
mới trộn đúng vào index 1024-chiều cosine hiện có.

---

## Bước 1 — LOCAL: export
```bash
# venv Chatbot, OpenSearch đang bật
PYTHONUTF8=1 ../Chatbot/Scripts/python.exe -m scripts.export_to_embed
#  → data/colab_export/{to_embed.jsonl, parents.jsonl, manifest.json}
# (Muốn làm cả 2012-2018 / toàn bộ 503: thêm --min-year 2012)
```

## Bước 2 — COLAB: embed trên GPU
1. Mở Colab → **Runtime → Change runtime type → GPU (T4 đủ dùng)**.
2. Upload `to_embed.jsonl` (41 MB) — **nên dùng Google Drive** (files.upload chậm/dễ
   lỗi với file lớn). Tải `to_embed.jsonl` vào Drive thư mục `MyDrive/legalai/`.
3. Chạy cell sau (đọc/ghi qua Drive — file ra ~250 MB nên KHÔNG dùng files.download):

```python
!pip -q install -U sentence-transformers pyarrow
from google.colab import drive; drive.mount("/content/drive")
D = "/content/drive/MyDrive/legalai"          # nơi đặt to_embed.jsonl

import json, torch, numpy as np
import pyarrow as pa, pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer

rows = [json.loads(l) for l in open(f"{D}/to_embed.jsonl", encoding="utf-8")]
texts = [r["text"] for r in rows]
print(len(texts), "chunks")

# KHỚP src/embedding.Embedder: BGE-M3, fp16 GPU, max_seq 1024, normalize, float32
model = SentenceTransformer("BAAI/bge-m3", device="cuda",
                            model_kwargs={"torch_dtype": torch.float16})
model.max_seq_length = 1024
emb = model.encode(texts, batch_size=64, normalize_embeddings=True,
                   convert_to_numpy=True, show_progress_bar=True)
emb = emb.astype(np.float32)            # quan trọng: cast về float32

for r, e in zip(rows, emb):
    r["embedding"] = e.tolist()

pq.write_table(pa.Table.from_pylist(rows), f"{D}/embedded.parquet")
print("Saved:", len(rows), "rows →", f"{D}/embedded.parquet")
```

> ~48.681 chunk → trên T4 khoảng **~10-20 phút**. `batch_size=64` cho T4; A100/L4
> để `128`. Nếu OOM → giảm batch. File `embedded.parquet` ~250 MB → tải từ Drive về.
> 4. Tải `embedded.parquet` từ Drive về máy, đặt vào `data/colab_export/`.

## Bước 3 — LOCAL: merge vào OpenSearch
```bash
# đặt embedded.parquet vào data/colab_export/
PYTHONUTF8=1 ../Chatbot/Scripts/python.exe -m scripts.merge_embedded --dry-run  # kiểm tra
PYTHONUTF8=1 ../Chatbot/Scripts/python.exe -m scripts.merge_embedded            # ghi thật
# → xoá chunk cũ + add chunk mới (có vector) + ghi parent_store.db
```

## Bước 4 — xác nhận
```bash
PYTHONUTF8=1 ../Chatbot/Scripts/python.exe -m scripts.audit_coverage
PYTHONUTF8=1 ../Chatbot/Scripts/python.exe -m scripts.scan_unstructured  # số VB còn thiếu giảm mạnh
# sau đó nên fstrim WSL2 (VHDX phình do merge churn)
```

---
### Vì sao ghép được (tóm tắt)
- Cùng model BGE-M3 → vector **giống hệt** như embed local → tương thích index.
- chunk_id giữ scheme cũ (prefix theo source) → nhất quán; merge xoá doc cũ trước
  khi add nên không trùng/lẫn.
- parent text ghi vào `parent_store.db` (SQLite local), không nằm trong OpenSearch.
- Đã trừ sẵn các VB embed bằng CPU trước đó (đọc batch_reingest_progress.json).
