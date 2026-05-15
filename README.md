# ProjectGenAI_2: Legal AI System

**Legal AI System** là một hệ thống trí tuệ nhân tạo hỗ trợ tư vấn và phân tích pháp luật, được thiết kế để:

* Trả lời câu hỏi pháp lý
* Phân tích bản án
* Đánh giá tính hợp lý của sự việc
* Hỗ trợ nghiên cứu pháp luật

Hệ thống là sự kết hợp chặt chẽ giữa:

* **Large Language Model (LLM)** * **Vector Retrieval (Semantic Search)** * **Knowledge Graph (Logical Reasoning)** > **→ Tạo thành một kiến trúc Graph-RAG (Retrieval-Augmented Generation nâng cao).**

---

## Tính năng chính

### 1. Hỏi đáp pháp luật

* Trả lời câu hỏi bằng ngôn ngữ tự nhiên.
* Trích dẫn điều luật cụ thể, chính xác.

### 2. Phân tích bản án

* Tóm tắt nội dung bản án.
* Xác định các điều luật được áp dụng.
* Phân tích logic pháp lý của vụ việc.

### 3. Kiểm tra tính hợp lý

* Phát hiện các điểm bất hợp lý trong sự việc.
* So sánh, đối chiếu với các quy định của pháp luật.

### 4. Hỗ trợ nghiên cứu

* Tìm kiếm các văn bản luật liên quan.
* Tìm kiếm và gợi ý các án lệ tương tự.

---

## Luồng xử lý chi tiết

Dưới đây là quy trình hoạt động của hệ thống khi người dùng đặt câu hỏi:

**1. Query Understanding (Hiểu truy vấn)**

* Xác định intent (ý định của người dùng).
* Trích xuất các entity (thực thể) pháp lý.

**2. Graph Retrieval (Truy xuất đồ thị)**

* Tìm các node liên quan trong Knowledge Graph.
* Mở rộng context (ngữ cảnh) thông qua các relationship (mối quan hệ).

**3. Vector Retrieval (Truy xuất Vector)**

* Tìm các đoạn văn bản liên quan nhất thông qua tìm kiếm ngữ nghĩa.

**4. Fusion & Reranking (Dung hợp & Xếp hạng lại)**

* Kết hợp kết quả từ Graph và Vector.
* Lọc và loại bỏ các thông tin trùng lặp.

**5. Context Assembly (Tập hợp ngữ cảnh)**

* Graph summary (Tóm tắt thông tin từ đồ thị).
* Relevant chunks (Các đoạn văn bản liên quan nhất).
* Metadata (Thông tin đi kèm như: nguồn, tình trạng hiệu lực).

**6. LLM Generation (Sinh văn bản)**

* Tổng hợp và sinh câu trả lời cuối cùng.
* Trích dẫn các điều luật minh chứng.



## Điểm nổi bật (Highlights)

* Kết hợp Graph + Vector (Graph-RAG)
* Có khả năng reasoning pháp lý (có kiểm soát)
* Hỗ trợ phân tích bản án thực tế
* Có thể mở rộng thành Legal AI Assistant

---

## Giai đoạn hiện tại: RAG đơn giản

Pipeline khởi đầu (trước khi thêm Knowledge Graph):

```
Raw legal docs (PDF/HTML)
    -> Parsing + Cleaning
    -> Chunking (theo Điều/Khoản/Điểm)
    -> Embedding
    -> Vector DB (Chroma)
    -> Retriever
    -> LLM (sinh câu trả lời + trích dẫn nguồn)
```

### Cấu trúc thư mục

```
ProjectGenAI_2/
├── data/
│   ├── raw/          # PDF/HTML gốc (gitignored)
│   ├── processed/    # JSON sau chunking (gitignored)
│   └── vectorstore/  # Chroma persistent dir (gitignored)
├── src/
│   ├── schemas.py    # Pydantic models: Chunk, RawDocument, Answer, ...
│   ├── config.py     # Đọc .env, đường dẫn, hyperparams
│   ├── parsing.py    # PDF/HTML -> text sạch
│   ├── chunking.py   # Chunk theo cấu trúc pháp lý
│   ├── embedding.py  # Wrapper BGE-M3 / sentence-transformers
│   ├── vectorstore.py# Wrapper Chroma
│   ├── retriever.py  # Top-k + (sau này) BM25 hybrid + rerank
│   └── generator.py  # LLM Claude + prompt + citation parsing
├── scripts/ingest.py # Pipeline ingestion end-to-end
├── tests/
├── app.py            # CLI hỏi-đáp
├── requirements.txt
└── .env.example
```

### Cài đặt

```powershell
# 1. Tạo virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài dependencies
pip install -r requirements.txt

# 3. Config
Copy-Item .env.example .env
# rồi điền ANTHROPIC_API_KEY vào .env
```

### Sử dụng

```powershell
# Bỏ PDF/HTML vào data/raw/, sau đó:
python -m scripts.ingest    # parse + chunk + embed + lưu vector store
python app.py               # CLI hỏi-đáp tương tác
```

### Trạng thái

Hiện tại các module trong `src/` đều là **stub** (`raise NotImplementedError`).
Implement theo thứ tự: `parsing` -> `chunking` -> `embedding` -> `vectorstore`
-> `retriever` -> `generator`. Mỗi bước nên có 1 test ở `tests/` xác nhận
chạy được trên 1 file mẫu trước khi sang bước kế.
