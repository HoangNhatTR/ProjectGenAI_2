"""Chạy bộ câu hỏi test (tests/demo_questions.json) qua API và xuất báo cáo.

    python -m scripts.run_question_suite                      # chạy đủ 77 câu (~45-90 phút)
    python -m scripts.run_question_suite --limit 5            # smoke test nhanh
    python -m scripts.run_question_suite --category GT,CALC   # chỉ vài nhóm
    python -m scripts.run_question_suite --model legal-ai-full

Kết quả:
    data/question_suite_report.md   — bảng PASS/MISS + thời gian từng câu
    data/question_suite_raw.jsonl   — câu trả lời đầy đủ (1 JSON/dòng) để soi lại

PASS = câu trả lời chứa ít nhất 1 từ khóa trong expect_any (không phân biệt
hoa thường). MISS không có nghĩa là sai — chỉ là cần người đọc soi lại.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = ROOT / "tests" / "demo_questions.json"
DEFAULT_REPORT = ROOT / "data" / "question_suite_report.md"
DEFAULT_RAW = ROOT / "data" / "question_suite_raw.jsonl"


def ask(base_url: str, model: str, question: str, history: list[dict], timeout: int) -> tuple[str, float]:
    messages = list(history) + [{"role": "user", "content": question}]
    t0 = time.time()
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json={"model": model, "messages": messages, "stream": False},
        timeout=timeout,
    )
    dt = time.time() - t0
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"], dt


def main() -> None:
    p = argparse.ArgumentParser(description="Chạy bộ câu hỏi test qua API")
    p.add_argument("--file", default=str(DEFAULT_FILE))
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="legal-ai-graph")
    p.add_argument("--category", default=None, help="Lọc nhóm, vd: GT,CALC,KG")
    p.add_argument("--limit", type=int, default=None, help="Chỉ chạy N câu đầu (smoke)")
    p.add_argument("--delay", type=float, default=2.0, help="Giây nghỉ giữa các câu (né rate limit LLM)")
    p.add_argument("--timeout", type=int, default=300, help="Timeout mỗi câu (giây)")
    args = p.parse_args()

    data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    questions = data["questions"]
    if args.category:
        cats = {c.strip().upper() for c in args.category.split(",")}
        questions = [q for q in questions if q["category"].upper() in cats]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("Không có câu nào khớp bộ lọc.")
        return

    # API sống không?
    health = requests.get(f"{args.base_url}/", timeout=10).json()
    print(f"API OK — {health.get('chunks', '?'):,} chunks | model={args.model} | {len(questions)} câu")

    rows: list[dict] = []
    raw_path = Path(DEFAULT_RAW)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_f = raw_path.open("w", encoding="utf-8")

    t_suite = time.time()
    for i, q in enumerate(questions, 1):
        try:
            answer, dt = ask(args.base_url, args.model, q["question"],
                             q.get("history", []), args.timeout)
            low = answer.lower()
            hits = [k for k in q.get("expect_any", []) if k.lower() in low]
            status = "PASS" if hits else "MISS"
            err = ""
        except Exception as exc:
            answer, dt, hits, status, err = "", 0.0, [], "ERROR", str(exc)[:200]

        rows.append({
            "id": q["id"], "category": q["category"], "question": q["question"],
            "status": status, "secs": round(dt, 1),
            "hits": hits, "answer_head": answer[:150].replace("\n", " "),
            "error": err,
        })
        raw_f.write(json.dumps({**q, "answer": answer, "secs": round(dt, 1),
                                "status": status, "error": err},
                               ensure_ascii=False) + "\n")
        raw_f.flush()
        print(f"[{i}/{len(questions)}] {q['id']} {status} ({dt:.0f}s) {q['question'][:60]}")
        if i < len(questions):
            time.sleep(args.delay)
    raw_f.close()

    # ── Báo cáo Markdown ──────────────────────────────────────────────────────
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    n_err = sum(1 for r in rows if r["status"] == "ERROR")
    total_min = (time.time() - t_suite) / 60
    by_cat: dict[str, list[dict]] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)

    lines = [
        f"# Báo cáo test chatbot — {time.strftime('%Y-%m-%d %H:%M')}",
        "",
        f"- Model: `{args.model}` @ {args.base_url}",
        f"- Kết quả: **{n_pass}/{len(rows)} PASS** ({n_err} lỗi) — {total_min:.0f} phút",
        f"- Trung bình: {sum(r['secs'] for r in rows) / max(len(rows), 1):.0f}s/câu",
        "",
        "## Theo nhóm",
        "",
        "| Nhóm | PASS | Tổng |",
        "|---|---|---|",
    ]
    for cat, rs in sorted(by_cat.items()):
        lines.append(f"| {cat} | {sum(1 for r in rs if r['status'] == 'PASS')} | {len(rs)} |")
    lines += ["", "## Chi tiết", "",
              "| ID | KQ | Giây | Câu hỏi | Trả lời (đầu) |", "|---|---|---|---|---|"]
    for r in rows:
        mark = {"PASS": "✅", "MISS": "⚠️", "ERROR": "❌"}[r["status"]]
        detail = r["error"] if r["status"] == "ERROR" else r["answer_head"]
        lines.append(f"| {r['id']} | {mark} | {r['secs']} | {r['question'][:60]} | {detail[:100]} |")

    report = Path(DEFAULT_REPORT)
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{'='*60}\nPASS {n_pass}/{len(rows)} | lỗi {n_err} | {total_min:.0f} phút")
    print(f"Báo cáo : {report}")
    print(f"Raw     : {raw_path}")


if __name__ == "__main__":
    main()
