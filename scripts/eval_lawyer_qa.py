"""Benchmark CHẤT LƯỢNG TRẢ LỜI Module 1 vs LLM khác — KHÔNG chấm tự động.

Khác scripts/eval_retrieval.py (chỉ đo retrieve, tất định, hit@k): script này
GỌI THẬT hệ thống (Module 1 /v1/chat/completions, model mặc định frontend
"legal-ai-graph") + N LLM benchmark RAW (không RAG, gọi thẳng qua KieAIClient)
cho cùng bộ câu hỏi, rồi LOG song song để người/AI đọc chấm ngữ nghĩa sau
(rubric docs/tro-ly-luat-su/template-danh-gia-ket-qua.md mục 3). Script chỉ
tính các chỉ số TẤT ĐỊNH quan sát được (latency, độ dài, có trích dẫn không,
có dấu hiệu từ chối không) — KHÔNG tự chấm đúng/sai nội dung pháp lý.

Resumable: ghi từng câu vào results.jsonl ngay sau khi xong (append), file
progress.json lưu id đã xong — chạy lại chỉ làm nốt phần còn thiếu.

    cd ProjectGenAI_2
    ../Chatbot/Scripts/python.exe -m scripts.eval_lawyer_qa --limit 20 --tag pilot
    ../Chatbot/Scripts/python.exe -m scripts.eval_lawyer_qa --tag full --concurrency 4
    ../Chatbot/Scripts/python.exe -m scripts.eval_lawyer_qa --resume --tag full
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Optional

import requests

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GOLD = ROOT / "tests" / "eval_lawyer_qa_gold.json"
REPORT_DIR = ROOT / "data" / "eval_lawyer_qa"

MODULE1_URL = "http://localhost:8000/v1/chat/completions"
MODULE1_MODEL = "legal-ai-graph"  # mặc định thật của frontend (lib/store.ts)
MODULE1_TIMEOUT_S = 240

BENCHMARK_SYSTEM_PROMPT = (
    "Bạn là trợ lý tư vấn pháp luật Việt Nam, có kiến thức về hệ thống văn bản "
    "quy phạm pháp luật. Hãy trả lời câu hỏi sau của người dùng một cách chính "
    "xác và đầy đủ nhất theo hiểu biết của bạn, trích dẫn điều luật/văn bản cụ "
    "thể nếu bạn biết rõ (đừng bịa số điều nếu không chắc)."
)

REFUSAL_MARKERS = (
    "tôi không thể", "xin lỗi, tôi không", "không thể hỗ trợ",
    "vượt quá khả năng", "không thể trả lời", "ngoài phạm vi",
    "vui lòng cung cấp thêm", "bạn có thể cho biết thêm", "cần thêm thông tin",
)


def call_module1(question: str, model: str = MODULE1_MODEL) -> dict[str, Any]:
    t0 = time.time()
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": question}],
        "stream": False,
    }
    try:
        resp = requests.post(MODULE1_URL, json=payload, timeout=MODULE1_TIMEOUT_S)
        dt = time.time() - t0
        if resp.status_code != 200:
            return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:300]}", "secs": round(dt, 1)}
        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        content = ((choice.get("message") or {}).get("content")) or ""
        return {
            "ok": True,
            "answer": content,
            "citations": data.get("legal_citations"),
            "confidence": data.get("confidence"),
            "secs": round(dt, 1),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "secs": round(time.time() - t0, 1)}


def call_benchmark(question: str, model_slug: str, kie_key: str, kie_host: str) -> dict[str, Any]:
    from src.llm_client import KieAIClient

    t0 = time.time()
    try:
        client = KieAIClient(api_key=kie_key, base_url=kie_host)
        resp = client.chat(
            model=model_slug,
            messages=[
                {"role": "system", "content": BENCHMARK_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            options={"temperature": 0.2},
        )
        dt = time.time() - t0
        content = (resp.get("message") or {}).get("content") or ""
        return {"ok": True, "answer": content, "secs": round(dt, 1)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "secs": round(time.time() - t0, 1)}


def _looks_like_refusal(answer: str) -> bool:
    low = answer.lower()
    return any(m in low for m in REFUSAL_MARKERS)


def _has_citation(answer: str) -> bool:
    low = answer.lower()
    return ("điều " in low) or ("[1]" in answer) or ("khoản " in low)


def process_one(q: dict, benchmark_models: list[str], kie_key: str, kie_host: str) -> dict:
    """Gọi hệ thống + mọi benchmark model SONG SONG cho 1 câu (độc lập nhau —
    system chạm localhost:8000, benchmark chạm kie.ai — không tranh CPU đáng kể)."""
    result: dict[str, Any] = {"id": q["id"], "category": q["category"], "question": q["question"],
                              "difficulty": q.get("difficulty"), "note": q.get("note")}
    with ThreadPoolExecutor(max_workers=1 + len(benchmark_models)) as inner:
        fut_system = inner.submit(call_module1, q["question"])
        fut_bench = {slug: inner.submit(call_benchmark, q["question"], slug, kie_key, kie_host)
                     for slug in benchmark_models}
        result["system"] = fut_system.result()
        result["benchmarks"] = {slug: f.result() for slug, f in fut_bench.items()}
    return result


def main() -> None:
    p = argparse.ArgumentParser(description="Benchmark chất lượng trả lời Module 1 vs LLM raw")
    p.add_argument("--gold", default=str(DEFAULT_GOLD))
    p.add_argument("--category", default=None, help="Lọc nhóm, vd HS,GT")
    p.add_argument("--ids", default=None, help="Chọn đích danh id, phẩy cách, vd HS01,GT04")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--models", default="gemini-2.5-pro,gpt-5-2", help="Slug model benchmark, phẩy cách")
    p.add_argument("--concurrency", type=int, default=3)
    p.add_argument("--tag", required=True, help="Tên đợt chạy — quyết định tên file output")
    p.add_argument("--resume", action="store_true", help="Bỏ qua id đã có trong results.jsonl cùng tag")
    args = p.parse_args()

    from src import config
    from src.pipeline import provider_credentials
    kie_key, kie_host = provider_credentials("kieai")
    benchmark_models = [m.strip() for m in args.models.split(",") if m.strip()]

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    questions = gold["questions"]
    if args.category:
        cats = {c.strip().upper() for c in args.category.split(",")}
        questions = [q for q in questions if q["category"].upper() in cats]
    if args.ids:
        wanted = [i.strip().upper() for i in args.ids.split(",") if i.strip()]
        by_id = {q["id"].upper(): q for q in questions}
        questions = [by_id[i] for i in wanted if i in by_id]
    if args.limit:
        questions = questions[: args.limit]

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = REPORT_DIR / f"results_{args.tag}.jsonl"
    done_ids: set[str] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done_ids.add(json.loads(line)["id"])
        questions = [q for q in questions if q["id"] not in done_ids]

    print(f"[eval] {len(questions)} câu (đã xong {len(done_ids)}) | benchmark={benchmark_models} | "
          f"concurrency={args.concurrency} | Module1={MODULE1_URL} model={MODULE1_MODEL}")

    mode = "a" if (args.resume and results_path.exists()) else "w"
    t_start = time.time()
    n_done = 0
    with open(results_path, mode, encoding="utf-8") as fh, \
         ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(process_one, q, benchmark_models, kie_key, kie_host): q
            for q in questions
        }
        for fut in as_completed(futures):
            q = futures[fut]
            try:
                row = fut.result()
            except Exception as exc:
                row = {"id": q["id"], "category": q["category"], "question": q["question"], "error": str(exc)}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            n_done += 1
            sys_secs = (row.get("system") or {}).get("secs", "?")
            sys_ok = (row.get("system") or {}).get("ok")
            print(f"  [{n_done:>3}/{len(questions)}] {row['id']:<8} system_ok={sys_ok} "
                  f"({sys_secs}s) {row['question'][:55]}")

    print(f"\n[eval] Xong {n_done} câu trong {time.time()-t_start:.0f}s → {results_path}")
    _write_deterministic_report(results_path, args.tag, benchmark_models)


def _write_deterministic_report(results_path: Path, tag: str, benchmark_models: list[str]) -> None:
    """Tổng hợp chỉ số TẤT ĐỊNH (latency, tỉ lệ có trích dẫn, tỉ lệ từ chối) —
    KHÔNG chấm đúng/sai ngữ nghĩa (việc đó do người/AI đọc log rồi chấm rubric riêng)."""
    rows = [json.loads(l) for l in results_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    systems = ["system"] + benchmark_models

    def stats_for(key: str) -> dict:
        secs, ok_n, cite_n, refusal_n, n = [], 0, 0, 0, 0
        for r in rows:
            entry = r.get("system") if key == "system" else (r.get("benchmarks") or {}).get(key)
            if not entry:
                continue
            n += 1
            if entry.get("ok"):
                ok_n += 1
                secs.append(entry.get("secs", 0))
                ans = entry.get("answer", "")
                if _has_citation(ans):
                    cite_n += 1
                if _looks_like_refusal(ans):
                    refusal_n += 1
        secs.sort()
        p50 = secs[len(secs) // 2] if secs else 0
        p95 = secs[int(len(secs) * 0.95)] if secs else 0
        return {"n": n, "ok": ok_n, "avg_secs": round(sum(secs) / len(secs), 1) if secs else 0,
                "p50_secs": p50, "p95_secs": p95, "cite_rate": round(cite_n / ok_n, 2) if ok_n else 0,
                "refusal_rate": round(refusal_n / ok_n, 2) if ok_n else 0}

    summary = {s: stats_for(s) for s in systems}
    lines = ["# Eval Lawyer QA — báo cáo chỉ số tất định", "",
              f"> Đợt: **{tag}** | số câu: **{len(rows)}** | "
              "chấm ngữ nghĩa (rubric) làm RIÊNG, xem file kèm theo.", "",
              "| hệ thống | n | ok | avg(s) | p50(s) | p95(s) | %có trích dẫn | %giống từ chối |",
              "|---|---|---|---|---|---|---|---|"]
    for s, m in summary.items():
        lines.append(f"| {s} | {m['n']} | {m['ok']} | {m['avg_secs']} | {m['p50_secs']} | "
                      f"{m['p95_secs']} | {m['cite_rate']*100:.0f}% | {m['refusal_rate']*100:.0f}% |")
    report_path = REPORT_DIR / f"report_{tag}.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[eval] report → {report_path}")
    for s, m in summary.items():
        print(f"  {s:<16} n={m['n']:<4} ok={m['ok']:<4} avg={m['avg_secs']:>6}s "
              f"p95={m['p95_secs']:>6}s cite%={m['cite_rate']*100:>4.0f} refusal%={m['refusal_rate']*100:>4.0f}")


if __name__ == "__main__":
    main()
