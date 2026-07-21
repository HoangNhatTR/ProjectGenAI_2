# Changelog

---

## Citation refs + Latency — 2026-07-07

### Sửa lỗi trích dẫn (popup UI trống)

Thân bài LLM trích `[3]`, `[5]` theo số thứ tự context, nhưng block
"📚 Nguồn pháp lý" đánh số lại từ `[1]` → UI (legal-chat-ui) tra map thất bại,
popup chỉ hiện "Nguồn 5".

- `src/schemas.py` — `Citation` thêm `ref` (số [n] trong thân bài) + `title`
- `src/generator.py` — gom `_build_citations()` dùng chung cho generate/stream,
  gắn `ref` = index gốc 1-based của context
- `api.py` — `_format_citations` in `[ref]` thay vì đánh số lại

### Popup trích dẫn hiện TOÀN VĂN điều luật

Backend gửi thêm citations structured (field `legal_citations` — toàn văn đoạn
trích, giữ xuống dòng) trong SSE chunk cuối stream + response non-stream
(client OpenAI chuẩn bỏ qua field lạ). Block text "📚 Nguồn pháp lý" rút gọn
snippet về 240 ký tự cho gọn (toàn văn nằm ở popup).

- `api.py` — `_citations_payload()` + `_citations_chunk()` (stream) +
  `ChatCompletionResponse.legal_citations` (non-stream)
- `legal-chat-ui` — `types.ts` (`LegalCitation`, `Message.citations`),
  `api.ts` (callback `onCitations`), `store.ts` (`setAssistantCitations`),
  `InputBox.tsx` (wire), `MessageItem.tsx` (popup: tên văn bản + chip
  Điều/Khoản + toàn văn `whitespace-pre-wrap`, cuộn max-h-72; fallback parse
  block text cho message cũ)

### Giảm latency

- `src/pipeline.py` — `_retrieve_rerank`: parent expansion chuyển ra SAU rerank.
  CrossEncoder chấm child chunk ~600 ký tự thay vì full Điều đã expand (hàng
  nghìn ký tự) — khâu từng tốn 1.5–5s/câu trên CPU. Rerank cả pool → expand
  (dedup theo Điều) → cắt top_k nên vẫn đủ top_k Điều phân biệt.
- `src/reranker.py` — `CE_MAX_CHARS` (mặc định 2000): chặn text quá dài vào CE,
  lưới an toàn cho caller khác (app.py/ui_app.py vẫn expand trước rerank)
- `api.py` — warmup nền lúc khởi động: preload CrossEncoder (~8s) + embedder,
  query đầu tiên không còn trả giá load model (đo được rerank đầu tốn 20s+)

---

## Data Supplement — 2026-06-10

### Dữ liệu mới

| Folder | Số file | Nội dung |
|---|---|---|
| `all_laws/` | 615 (+6) | 7 luật còn thiếu + luật mới 2025 |
| `nghi_dinh/` | ~200+ | Nghị định & Thông tư 9 lĩnh vực |
| `nghi_quyet/` | 3 | Nghị quyết HĐTP |
| `an_le/` | đang crawl | Án lệ TAND Tối cao |

### Cải tiến pipeline

- `src/vbpl_client.py` — Client mới 2 bước (search → GET /doc/{id}) thay cho docAbs cũ
- `scripts/crawl_nghi_dinh.py` — Crawler NĐ/TT 9 lĩnh vực
- `scripts/crawl_an_le.py` — Crawler Án lệ (toaan.gov.vn + vbpl.vn)
- `scripts/run_supplement.py` — Pipeline P1+P2 tổng hợp
- `scripts/post_crawl.py` — Enrich → Ingest → BM25 → Analyze
- `scripts/check_missing_laws.py` — Kiểm tra 74 luật quan trọng
- `scripts/analyze_data.py` — Thống kê Chương/Mục/Điều/Khoản/Điểm
- `scripts/enrich_metadata.py` — Thêm LINH_VUC, VAN_BAN_THAY_THE...
- `src/schemas.py` — DocumentMetadata: thêm `linh_vuc`, `co_quan`, `folder`
- `scripts/ingest.py` — Xử lý file root + thư mục mới + metadata mới
- `data/raw/relationship_map.json` — Bản đồ quan hệ 419 văn bản

### Thống kê dữ liệu (toàn bộ sau supplement)

```
                  TRƯỚC       SAU
Văn bản         :   612     → 1,148 files (+87%)
Dung lượng      :  53 MB   → 134 MB   (+153%)
Chương          : 4,397    → 8,980
Mục             : 2,786    → 5,499
Điều            : 66,146   → 140,734  (+113%)
Khoản ~         : 153,319  → 298,611  (+95%)
Điểm  ~         : 85,449   → 186,300  (+118%)
Tổng QPPL       : ~305,000 → 625,645  (+2x)
```

**Phân bổ loại văn bản:**
- Luật/Bộ luật/Hiến pháp : 617 files (all_laws + root)
- Nghị định & Thông tư   : 440 files (nghi_dinh/)
- Án lệ                  :  91 files (an_le/)
- Nghị quyết HĐTP        :   3 files (nghi_quyet/)

**Lĩnh vực phủ rộng nhất:**
- Dân sự   : 260 VB | 32,375 Điều
- DN/ĐT    : 172 VB | 17,438 Điều
- Hình sự  :  77 VB | 12,904 Điều
- Hành chính: 107 VB | 10,164 Điều

---

## Big Update — Legal AI Agent

**Ngày cập nhật:** 2026-05-13  
**Phiên bản trước:** RAG Chatbot (hỏi đáp đơn giản)  
**Phiên bản hiện tại:** Legal AI Agent (đa công cụ, đa bước lý luận)

---

## Tổng quan kiến trúc mới

```
User Input
  │
  ▼
ConversationState.update_from_question()      ← NEW
  │
  ▼
SmartRouter  (intent detection + query rewrite)  ← UPGRADED
  │
  ├─ answer_direct ──────────────────────────► Trả thẳng (chitchat / meta / clarify)
  │
  ├─ use_tool ───────────────────────────────► LegalToolRegistry.execute()
  │                                                    │
  │                                                    ▼
  │                                            Generator.generate(tool_results)
  │
  └─ retrieve ──► LegalPlanner.create_plan()   ← NEW
                        │
                        ├─ simple ──► Retriever → Reranker → Generator
                        │
                        └─ complex ─► Tool steps (calculate / draft / lookup)
                                          │
                                          ▼
                                      Retriever → Reranker → Generator
                                                                  │
                                                                  ▼
                                                          Guardrails        ← NEW
                                                                  │
                                                                  ▼
                                                     ConversationState.update_from_answer()
                                                     Session + Memory update
```

---

## Các thành phần mới (5 file)

### 1. `src/state.py` — Conversation State Management

**Mục đích:** Ghi nhớ ngữ cảnh pháp lý xuyên suốt hội thoại để hiểu câu hỏi mơ hồ.

**Ví dụ:**
```
Lượt 1: "Vượt đèn đỏ bằng xe máy bị phạt bao nhiêu?"
Lượt 2: "Thế ô tô thì sao?"
         → State biết: topic=vi phạm giao thông, act=vượt đèn đỏ
         → Router rewrite: "Mức phạt vượt đèn đỏ đối với ô tô là bao nhiêu?"
```

**Thông tin được track:**
| Trường | Ví dụ |
|--------|-------|
| `current_legal_topic` | "vi phạm giao thông" |
| `current_act` | "vượt đèn đỏ" |
| `vehicle_type` | "xe máy" |
| `last_retrieved_law` | "Nghị định 100/2019/NĐ-CP" |
| `law_references` | ["Điều 6", "100/2019/NĐ-CP"] |

**Tự động cập nhật** sau mỗi câu hỏi và câu trả lời.  
**Reset** khi dùng lệnh `/clear`.

---

### 2. `src/tools.py` — Tool Calling Framework

**Mục đích:** Cung cấp công cụ đặc biệt cho Agent, vượt qua giới hạn của RAG thuần.

**4 tools hiện có:**

| Tool | Mô tả | Khi dùng |
|------|--------|----------|
| `legal_search` | Tìm văn bản pháp luật liên quan | Tra cứu chủ đề rộng |
| `law_article_lookup` | Tra Điều/Khoản cụ thể (vd: "Điều 6 NĐ 100") | Khi hỏi đúng điều khoản |
| `calculate_fine` | Tính tổng tiền phạt nhiều lỗi (RAG + LLM tính) | "Bị phạt tổng bao nhiêu?" |
| `draft_document` | Soạn đơn, công văn, hợp đồng mẫu | "Soạn đơn khiếu nại..." |

**Ví dụ calculate_fine:**
```
User: "Tôi vượt đèn đỏ và không mang GPLX, xe máy, bị phạt tổng bao nhiêu?"
→ Tool retrieve luật → LLM tính từng lỗi → trả bảng tổng kết
```

---

### 3. `src/planner.py` — Planner / Reasoning Layer

**Mục đích:** Phân tích câu hỏi phức tạp → chia thành bước nhỏ → thực thi tuần tự.

**2 loại plan:**

**Simple** (1 bước — RAG thông thường):
```
"Mức phạt vượt đèn đỏ là gì?"
→ Plan: [retrieve]
```

**Complex** (nhiều bước — dùng tools):
```
"Tôi vi phạm 3 lỗi: vượt đèn đỏ, không GPLX, nồng độ cồn. Phạt tổng bao nhiêu?"
→ Plan:
    Step 1: calculate_fine("vượt đèn đỏ xe máy")
    Step 2: calculate_fine("không mang GPLX xe máy")
    Step 3: calculate_fine("nồng độ cồn xe máy")
    Step 4: retrieve (tổng hợp)
→ Execute từng step → Generator tổng hợp kết quả
```

---

### 4. `src/reranker.py` — Legal Document Reranker

**Mục đích:** Sắp xếp lại kết quả retrieve — ưu tiên chunks khớp đúng Điều/Khoản trong câu hỏi.

**Lý do cần:** Vector search giỏi hiểu ngữ nghĩa nhưng có thể bỏ lỡ "Điều 6 Khoản 4" khi chunk không nằm gần nhau trong embedding space.

**Cách hoạt động:**
- Trích xuất tham chiếu từ query: `"Điều 6"`, `"100/2019/NĐ-CP"`
- Boost score (+0.20) cho chunk có chứa tham chiếu đó
- Boost nhỏ (+0.05) cho từ khóa pháp lý trùng khớp
- Sắp xếp lại → trả top-K tốt hơn

**Không cần tải model riêng** — rule-based, chạy nhanh.

---

### 5. `src/guardrails.py` — Legal AI Safety Guardrails

**Mục đích:** Kiểm soát an toàn câu trả lời pháp luật theo nguyên tắc:
- Không khẳng định thay luật sư
- Không bịa căn cứ
- Phân biệt thông tin pháp luật và tư vấn cá nhân
- Cảnh báo khi thiếu dữ kiện

**3 loại cảnh báo tự động:**

| Điều kiện | Hành động |
|-----------|-----------|
| Câu hỏi tình huống cá nhân ("tôi bị...", "của tôi...") | Disclaimer personal |
| Không có văn bản căn cứ trong DB | Warning thiếu căn cứ + hướng dẫn tra nguồn chính thức |
| Câu hỏi phức tạp (khởi kiện, phạt tù...) | Khuyến nghị gặp luật sư |

**Tắt guardrails:** `python app.py --no-guardrails`

---

## Các thành phần nâng cấp (3 file)

### `src/router.py` — Smart Router (UPGRADED)

**Thêm mới:**
- Nhận `ConversationState` → viết lại query có ngữ cảnh
- 4 intent mới: `compare`, `calculate`, `draft`, `followup`
- Action mới: `use_tool` (chỉ định thẳng tool cần gọi)

**Intents đầy đủ:**

| Intent | Mô tả | Action |
|--------|--------|--------|
| `legal` | Hỏi quy định/mức phạt cụ thể | retrieve |
| `consulting` | Tư vấn tình huống cá nhân | retrieve |
| `compare` | So sánh 2 trường hợp | retrieve |
| `calculate` | Tính tổng tiền phạt | **use_tool** |
| `draft` | Soạn đơn/văn bản | **use_tool** |
| `followup` | Câu hỏi tiếp theo có ngữ cảnh | retrieve (rewritten) |
| `chitchat` | Chào hỏi, cảm ơn | answer_direct |
| `meta` | Hỏi về chatbot | answer_direct |
| `clarify` | Câu hỏi quá mơ hồ | answer_direct (hỏi lại) |

---

### `src/generator.py` — Generator (UPGRADED)

**Thêm mới:**
- Tham số `tool_results: list[ToolResult]` — kết quả tools được chèn vào context
- Tham số `state_context: str` — ngữ cảnh conversation state trong system prompt
- Method `get_client()` — expose Ollama client cho Tools và Planner dùng chung

**Cấu trúc prompt mới:**
```
=== KẾT QUẢ TỪ TOOLS ===        (nếu có tool_results)
[Tool: calculate_fine | OK]
...kết quả tính toán...

=== NGỮ CẢNH VĂN BẢN PHÁP LUẬT ===
[1] Nguồn: nd100.txt - Điều 6 - Khoản 3
...nội dung chunk...

---
Câu hỏi: ...
```

---

### `app.py` — Main Application (UPGRADED)

**Thêm mới:**
- Import toàn bộ 5 module mới
- `ConversationState` khởi tạo per-session (in-memory)
- Pipeline 6 bước: State → Router → Planner → Tools → Reranker → Generator → Guardrails
- Lệnh `/state` mới: xem ngữ cảnh hội thoại hiện tại
- Flag `--no-planner`: tắt planner tiết kiệm 1 LLM call/lượt
- Flag `--no-guardrails`: tắt disclaimer
- UI rõ hơn: hiện `[intent=xxx]`, `[Planner] complex → N bước tool`, tool status `[OK]`/`[FAIL]`

---

## Lệnh mới trong CLI

```
/state     Xem Conversation State hiện tại
           (chủ đề, loại xe, luật đã tra, tham chiếu điều khoản)
```

**Flags mới khi khởi động:**
```bash
python app.py --no-planner       # Tắt planner (chỉ RAG thuần)
python app.py --no-guardrails    # Tắt disclaimer pháp lý
python app.py --no-memory-extract  # Tắt auto-extract memory (đã có từ trước)
```

---

## So sánh trước / sau

| Tính năng | Trước (RAG) | Sau (Legal AI Agent) |
|-----------|-------------|----------------------|
| Hiểu câu hỏi mơ hồ | Không | Có (Conversation State) |
| Intent detection | 5 loại | 9 loại |
| Query rewrite | Cơ bản | State-aware |
| Tool calling | Không | 4 tools |
| Tính tiền phạt tổng | Không | Có (calculate_fine) |
| Soạn văn bản | Không | Có (draft_document) |
| Tra Điều/Khoản cụ thể | Không | Có (law_article_lookup) |
| Lập kế hoạch đa bước | Không | Có (Planner) |
| Reranking | RRF fusion | RRF + Legal Reranker |
| An toàn pháp lý | Prompt cơ bản | Guardrails tự động |

---

## Roadmap tiếp theo (Giai đoạn 3)

- [ ] **Neural Reranker** — bge-reranker / cross-encoder thay rule-based
- [ ] **FastAPI Backend** — expose REST API cho frontend React/Streamlit
- [ ] **Evaluation Pipeline** — đánh giá độ chính xác câu trả lời
- [ ] **Web Search Tool** — kiểm tra luật mới từ vbpl.vn / thuvienphapluat.vn
- [ ] **Monitoring** — track query success rate, cost, latency
- [ ] **Knowledge Graph** — liên kết các điều khoản, văn bản với nhau
