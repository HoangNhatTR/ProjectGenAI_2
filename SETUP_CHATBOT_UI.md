# Hướng dẫn kết nối chatbot-ui với Legal AI Agent

## Kiến trúc

```
chatbot-ui (Next.js :3000)
        ↕  POST /v1/chat/completions
FastAPI backend (:8000)  ←→  api.py
        ↕
Legal AI Agent (RAG + KG + Ollama)
```

---

## Bước 1 — Cài và chạy FastAPI backend

```powershell
# Kích hoạt venv
.\Chatbot\Scripts\Activate.ps1

# Cài thêm fastapi + uvicorn
pip install fastapi "uvicorn[standard]" python-multipart

# Chạy API server
cd ProjectGenAI_2
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Kiểm tra hoạt động:

```
http://localhost:8000/           → health check
http://localhost:8000/v1/models  → danh sách models
http://localhost:8000/docs       → Swagger UI
```

---

## Bước 2 — Cài chatbot-ui

> **Yêu cầu**: Node.js ≥ 18 (https://nodejs.org)

### Option A: Phiên bản đơn giản (không cần Supabase) ✅ Khuyến nghị

```bash
# Clone nhánh stable cũ — chỉ cần OpenAI API key
git clone https://github.com/mckaywrigley/chatbot-ui.git
cd chatbot-ui
git checkout tags/v0.3.3 -b simple
npm install
```

Tạo file `.env.local`:

```env
# Trỏ vào FastAPI local thay vì api.openai.com
OPENAI_API_KEY=legal-ai-local
OPENAI_API_HOST=http://localhost:8000
```

```bash
npm run dev
# → Mở http://localhost:3000
```

### Option B: Phiên bản mới nhất (cần Supabase)

```bash
git clone https://github.com/mckaywrigley/chatbot-ui.git
cd chatbot-ui
npm install
```

Cần tạo project Supabase tại https://supabase.com và lấy:

- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `SUPABASE_SERVICE_ROLE_KEY`

Tạo `.env.local`:

```env
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJxxx...
SUPABASE_SERVICE_ROLE_KEY=eyJxxx...

# Legal AI Agent API
OPENAI_API_KEY=legal-ai-local
OPENAI_API_HOST=http://localhost:8000

# Tắt proxy để gọi thẳng vào API của mình
OPENAI_ORGANIZATION=
```

Chạy Supabase migration:

```bash
npx supabase db push
```

```bash
npm run dev
```

---

## Bước 3 — Chọn model trong chatbot-ui

Trong giao diện, chọn model:

| Model name         | Chế độ RAG                       |
| ------------------ | ----------------------------------- |
| `legal-ai-graph` | Graph-RAG (tốt nhất, mặc định) |
| `legal-ai-top15` | RAG chỉ top 15 luật (nhanh hơn)  |
| `legal-ai-full`  | RAG toàn bộ 609 luật             |
| `legal-ai`       | Alias của graph                    |

---

## Kiểm tra API bằng curl

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "legal-ai-graph",
    "messages": [{"role": "user", "content": "Vượt đèn đỏ xe máy phạt bao nhiêu?"}],
    "stream": false
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "legal-ai-graph",
    "messages": [{"role": "user", "content": "Vượt đèn đỏ xe máy phạt bao nhiêu?"}],
    "stream": true
  }' --no-buffer
```

---

## Thay thế khác (dễ hơn, không cần Node.js)

### Open WebUI với Docker

```bash
docker run -d \
  -p 3000:8080 \
  -e OPENAI_API_BASE_URL=http://host.docker.internal:8000/v1 \
  -e OPENAI_API_KEY=legal-ai-local \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

→ Mở http://localhost:3000 — giao diện ChatGPT, tự động nhận models từ `/v1/models`

### LibreChat

```bash
git clone https://github.com/danny-avila/LibreChat.git
cd LibreChat
cp .env.example .env
# Sửa .env:
# OPENAI_API_KEY=legal-ai-local
# OPENAI_REVERSE_PROXY=http://localhost:8000/v1
docker compose up
```

→ Mở http://localhost:3080

---

## Lưu ý

- FastAPI phải chạy **trước** khi mở chatbot-ui
- Ollama phải đang chạy (`ollama serve`) nếu dùng Ollama provider
- Citations được append vào cuối mỗi câu trả lời dạng Markdown
- Streaming: mỗi từ stream ~18ms (có thể chỉnh `WORD_STREAM_DELAY` trong `api.py`)
