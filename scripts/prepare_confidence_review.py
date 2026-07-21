"""Giảm công sức luật sư khi chấm gold cho P2.3 (calibration nhãn độ tin cậy).

KHÔNG tự chấm đúng/sai (cần đọc hiểu pháp lý — việc của luật sư), nhưng CHẠY
SẴN từng câu hỏi qua API thật rồi ghi câu trả lời + nhãn hệ thống ngay vào
tests/eval_confidence_gold.json (field system_answer_preview/system_confidence_label)
— luật sư chỉ cần ĐỌC ngay trong file, không phải tự chạy từng câu qua UI/curl
rồi lật lại gõ tay. Chỉ chạy cho câu CHƯA có graded_by (không đụng câu đã chấm).

    cd ProjectGenAI_2
    python -m scripts.prepare_confidence_review
    python -m scripts.prepare_confidence_review --gold tests/eval_confidence_gold.json --limit 5

Sau khi chạy: mở tests/eval_confidence_gold.json, đọc system_answer_preview +
system_confidence_label từng câu, điền "expected_correct": true/false và
"graded_by": "<tên>" — script eval_confidence.py sẽ tự nhận câu đã chấm.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

import requests

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = ROOT / "tests" / "eval_confidence_gold.json"


def main() -> None:
    p = argparse.ArgumentParser(description="Chạy sẵn câu hỏi P2.3 qua API + ghi preview vào gold JSON")
    p.add_argument("--gold", default=str(DEFAULT_GOLD))
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="legal-ai-graph")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--force", action="store_true", help="Chạy lại cả câu ĐÃ có graded_by (mặc định bỏ qua)")
    args = p.parse_args()

    gold_path = Path(args.gold)
    raw = json.loads(gold_path.read_text(encoding="utf-8"))
    questions = raw.get("questions", [])

    pending = [q for q in questions if args.force or not q.get("graded_by")]
    if args.limit:
        pending = pending[: args.limit]
    if not pending:
        print("[prepare_confidence_review] không có câu nào cần chạy "
              "(tất cả đã có graded_by — dùng --force để chạy lại).")
        return

    try:
        health = requests.get(f"{args.base_url}/", timeout=10).json()
    except Exception as exc:
        print(f"[prepare_confidence_review] Module 1 ({args.base_url}) không phản hồi: {exc}")
        return
    print(f"API OK — {health.get('chunks', '?'):,} chunks | {len(pending)} câu cần chạy")

    for i, q in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}] {q['id']}: {q['question'][:60]}")
        try:
            r = requests.post(
                f"{args.base_url}/v1/chat/completions",
                json={"model": args.model, "messages": [{"role": "user", "content": q["question"]}], "stream": False},
                timeout=args.timeout,
            )
            r.raise_for_status()
            resp = r.json()
        except Exception as exc:
            print(f"      LỖI: {exc}")
            continue
        answer_text = resp["choices"][0]["message"]["content"]
        conf = resp.get("confidence")
        q["system_answer_preview"] = answer_text[:800]
        q["system_confidence_label"] = conf["label_vi"] if conf else "(không áp dụng)"

    gold_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[prepare_confidence_review] đã ghi preview vào {gold_path}")
    print("Tiếp theo: mở file, đọc system_answer_preview từng câu, điền "
          "\"expected_correct\": true/false + \"graded_by\": \"<tên>\".")


if __name__ == "__main__":
    main()
