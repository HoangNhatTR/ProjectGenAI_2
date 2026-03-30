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
