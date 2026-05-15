# Big Update — Legal AI Agent

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
