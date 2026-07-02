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

## Kiến trúc hiện tại

```
                         ┌──────────────────────────────────────────┐
User ──► Router (LLM) ──►│ Flow A: answer_direct (chitchat/meta)    │
                         │ Flow B: tools (tính phạt, soạn văn bản,  │
                         │         so sánh, validate, KG lookup)    │
                         │ Flow C: Hybrid RAG                       │
                         │   Vector (BGE-M3 + Chroma)               │
                         │   + BM25  + Knowledge Graph (Neo4j)      │
                         │   → RRF fusion → CrossEncoder rerank     │
                         │   → Parent-child expansion (full Điều)   │
                         └────────────────┬─────────────────────────┘
                                          ▼
                         Generator (LLM) + Guardrails (disclaimer)
                                          ▼
                         Câu trả lời + 📚 trích dẫn Điều/Khoản/Điểm
```

Logic pipeline dùng chung nằm ở `src/pipeline.py` (`LegalPipeline`) —
cả API server lẫn UI đều gọi qua đây.

### Chế độ RAG

| Mode | Mô tả |
|---|---|
| `graph_rag` (mặc định) | Vector + BM25 + Knowledge Graph, top 15 luật |
| `rag_top10` | Vector + BM25, chỉ top 15 luật quan trọng (~7.5k chunks) |
| `rag_full` | Vector + BM25 trên toàn bộ 609 luật (~68k chunks) |

### Cấu trúc thư mục

```
ProjectGenAI_2/
├── data/                  # raw / processed / vectorstore / bm25 (gitignored)
├── src/
│   ├── pipeline.py        # LegalPipeline — logic chung cho mọi entry point
│   ├── config.py          # Đọc .env: model, provider, security
│   ├── schemas.py         # Pydantic: Chunk, Answer, Citation, ...
│   ├── parsing.py         # PDF/HTML/DOCX → text sạch
│   ├── chunking.py        # Chunk theo Điều/Khoản/Điểm + parent-child
│   ├── embedding.py       # BGE-M3
│   ├── vectorstore.py     # Chroma
│   ├── bm25_index.py      # BM25 (rank-bm25)
│   ├── retriever.py       # Hybrid: vector + BM25 + KG → RRF fusion
│   ├── reranker.py        # CrossEncoder bge-reranker (smart skip)
│   ├── router.py          # SmartRouter: intent → flow A/B/C
│   ├── tools.py           # calculate_fine, draft_document, compare, ...
│   ├── planner.py         # Multi-step plan cho câu hỏi phức tạp
│   ├── generator.py       # LLM generate + stream + citations
│   ├── guardrails.py      # Disclaimer pháp lý, cảnh báo thiếu căn cứ
│   ├── memory.py / session.py / state.py   # Hội thoại đa lượt
│   ├── llm_client.py      # Gemini/Groq/9Router/OpenRouter/KieAI clients
│   └── kg/                # Knowledge Graph: Neo4j extract + retrieve
├── scripts/               # ingest, crawler, build_bm25, build KG, benchmark
├── tests/                 # pytest — unit + endpoint tests
├── api.py                 # FastAPI — OpenAI-compatible API (streaming thật)
├── ui_app.py              # Streamlit UI (sessions, memory, KG explorer)
├── app.py                 # CLI hỏi-đáp
└── notebooks/             # Colab GPU embedding
```

### Cài đặt

```powershell
# 1. Tạo virtualenv (máy dev hiện tại đã có sẵn venv ..\Chatbot — dùng lại được:
#    ..\Chatbot\Scripts\Activate.ps1)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Cài dependencies (runtime + dev/test)
pip install -r requirements.txt -r requirements-dev.txt

# 3. Config
Copy-Item .env.example .env
# điền API key của provider muốn dùng (ROUTER9 / KIEAI / GEMINI / GROQ / OPENROUTER)
```

### Chạy

```powershell
# API server (OpenAI-compatible, dùng với legal-chat-ui / open-webui)
uvicorn api:app --host 0.0.0.0 --port 8000

# Streamlit UI đầy đủ tính năng
streamlit run ui_app.py

# CLI hỏi-đáp
python app.py

# Tests (cần requirements-dev.txt — pytest, httpx)
python -m pytest tests/ -q
```

### Bảo mật API

Hai biến env trong `.env` (xem `.env.example`):

- `API_AUTH_KEY` — nếu set, mọi request `/v1/*` phải gửi
  `Authorization: Bearer <key>`. Để trống = không bắt auth (chỉ dùng localhost).
- `CORS_ORIGINS` — danh sách origin được phép (mặc định localhost:3000/3001/8501).

### Ingest dữ liệu

```powershell
# Bỏ PDF/HTML vào data/raw/, sau đó:
python -m scripts.ingest        # parse + chunk + embed + lưu vector store
python -m scripts.build_bm25    # build BM25 index
# (tuỳ chọn) Knowledge Graph — cần Neo4j:
python -m scripts.build_structural_kg
python -m scripts.build_semantic_kg
```
