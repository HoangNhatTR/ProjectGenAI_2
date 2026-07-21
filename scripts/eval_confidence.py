"""Calibration cho nhãn độ tin cậy (P2.3, src/confidence.py) trên gold Q&A đã
người có chuyên môn CHẤM ĐÚNG/SAI thật — xem docs/tro-ly-luat-su/lo-trinh-cong-viec.md.

Script KHÔNG tự tạo gold (đúng/sai một câu trả lời pháp luật cần chuyên môn để
chấm, không phải việc script tự suy đoán được) — chỉ gọi API sống, đọc nhãn
`confidence` hệ thống đã gắn, rồi đối chiếu với `expected_correct` đã gán tay.
Câu hỏi thiếu `graded_by` bị BỎ QUA (không tính điểm) — tránh nhầm placeholder
thành gold thật, cùng nguyên tắc với scripts/eval_timeline.py.

Tiêu chí nghiệm thu roadmap: nhóm "Cao" phải đúng >=90%; nhóm "Thấp" được
PHÉP đúng <70% (nếu nhóm Thấp đúng ngang nhóm Cao thì nhãn vô dụng, không phân
biệt được gì).

    cd ProjectGenAI_2
    python -m scripts.eval_confidence --limit 5              # smoke nhanh
    python -m scripts.eval_confidence
    python -m scripts.eval_confidence --tag baseline
    python -m scripts.eval_confidence --compare baseline
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
DEFAULT_GOLD = ROOT / "tests" / "eval_confidence_gold.json"
REPORT_DIR = ROOT / "data" / "eval_confidence"


def ask(base_url: str, model: str, question: str, timeout: int) -> dict:
    r = requests.post(
        f"{base_url}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": question}], "stream": False},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.json()


def main() -> None:
    p = argparse.ArgumentParser(description="Calibration nhãn độ tin cậy (P2.3) trên gold Q&A đã chấm tay")
    p.add_argument("--gold", default=str(DEFAULT_GOLD))
    p.add_argument("--base-url", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="legal-ai-graph")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout", type=int, default=300)
    p.add_argument("--tag", default=None)
    p.add_argument("--compare", default=None)
    args = p.parse_args()

    raw = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    all_q = raw.get("questions", [])
    graded = [q for q in all_q if q.get("graded_by") and "expected_correct" in q]
    skipped = len(all_q) - len(graded)
    if args.limit:
        graded = graded[: args.limit]

    print(f"[eval_confidence] {len(graded)} câu đã chấm tay (bỏ qua {skipped} câu chưa có graded_by)")
    if not graded:
        print("Không có câu nào đủ điều kiện chấm điểm. Xem tests/eval_confidence_gold.json để điền gold.")
        return

    try:
        health = requests.get(f"{args.base_url}/", timeout=10).json()
    except Exception as exc:
        print(f"[eval_confidence] Module 1 ({args.base_url}) không phản hồi: {exc}")
        return
    print(f"API OK — {health.get('chunks', '?'):,} chunks | model={args.model}")

    rows = []
    for i, q in enumerate(graded, 1):
        try:
            resp = ask(args.base_url, args.model, q["question"], args.timeout)
        except Exception as exc:
            print(f"  [{i}/{len(graded)}] LỖI {q['id']}: {exc}")
            continue
        conf = resp.get("confidence")
        label = conf["label"] if conf else "khong_ap_dung"
        rows.append({
            "id": q["id"], "question": q["question"], "label": label,
            "expected_correct": bool(q["expected_correct"]),
        })
        print(f"  [{i:>2}/{len(graded)}] {q['id']:<10} label={label:<12} "
              f"expected_correct={q['expected_correct']}  {q['question'][:50]}")

    by_label: dict[str, list[dict]] = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    print("\n" + "=" * 60)
    print(f"{'nhãn':<14}{'n':>4}{'đúng':>6}{'accuracy':>10}")
    print("=" * 60)
    summary = {}
    for label, items in sorted(by_label.items()):
        n = len(items)
        n_correct = sum(1 for r in items if r["expected_correct"])
        acc = n_correct / n if n else 0.0
        summary[label] = {"n": n, "n_correct": n_correct, "accuracy": acc}
        print(f"{label:<14}{n:>4}{n_correct:>6}{acc:>10.0%}")

    cao = summary.get("cao", {"accuracy": None, "n": 0})
    thap = summary.get("thap", {"accuracy": None, "n": 0})
    print("\nTiêu chí nghiệm thu roadmap:")
    if cao["n"]:
        ok = cao["accuracy"] >= 0.90
        print(f"  Nhóm 'Cao' đúng >=90%: {'ĐẠT' if ok else 'CHƯA ĐẠT'} ({cao['accuracy']:.0%}, n={cao['n']})")
    else:
        print("  Nhóm 'Cao': chưa có mẫu nào trong gold")
    if thap["n"]:
        useful = thap["accuracy"] < (cao["accuracy"] if cao["n"] else 1.0)
        print(f"  Nhóm 'Thấp' đúng < nhóm 'Cao' (nhãn có phân biệt): "
              f"{'ĐẠT' if useful else 'CHƯA ĐẠT'} ({thap['accuracy']:.0%}, n={thap['n']})")
    else:
        print("  Nhóm 'Thấp': chưa có mẫu nào trong gold")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    snapshot = {"summary": summary, "rows": rows, "ts": time.time()}
    (REPORT_DIR / "latest.json").write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[eval_confidence] snapshot -> {REPORT_DIR / 'latest.json'}")

    if args.tag:
        (REPORT_DIR / f"snap_{args.tag}.json").write_text(
            json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval_confidence] đã lưu snapshot '{args.tag}'")

    if args.compare:
        snap_path = REPORT_DIR / f"snap_{args.compare}.json"
        if snap_path.exists():
            old = json.loads(snap_path.read_text(encoding="utf-8"))
            old_cao = old["summary"].get("cao", {}).get("accuracy")
            new_cao = summary.get("cao", {}).get("accuracy")
            print(f"\nSo sánh accuracy nhóm 'Cao': {old_cao} -> {new_cao}")
        else:
            print(f"\n[eval_confidence] không thấy snapshot: {snap_path}")


if __name__ == "__main__":
    main()
