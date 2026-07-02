"""Phase 1: Semantic KG extraction cho top N luật (mặc định: 10 luật quan trọng).

Pipeline:
  1. Query Neo4j: lấy danh sách Law cần extract (theo whitelist doc_number)
  2. Lấy text gốc từ file raw → parse Articles
  3. Mỗi Article → LLM extract Offense/Penalty/Subject + relations
  4. Bulk write vào Neo4j (MERGE idempotent)

Chạy:
    python -m scripts.build_semantic_kg                    # full top 10 (Gemini)
    python -m scripts.build_semantic_kg --skip-existing    # resume, skip Article đã có
    python -m scripts.build_semantic_kg --provider groq    # dùng Groq thay Gemini
    python -m scripts.build_semantic_kg --limit-articles 5 # test 5 điều/luật
    python -m scripts.build_semantic_kg --dry-run          # chỉ in, không write
    python -m scripts.build_semantic_kg --laws 100/2015/QH13,45/2019/QH14  # custom
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from src import config
from src.chunking import _iter_articles
from src.kg.neo4j_client import Neo4jClient
from src.kg.semantic_extractor import SemanticExtractor
from src.kg.structural_extractor import _normalize_for_kg, _article_id
from src.parsing import load_document
from scripts.ingest import iter_raw_files


# Top 15 luật demo-friendly (đời sống + đa lĩnh vực, ưu tiên phiên bản mới nhất)
DEFAULT_TOP_LAWS = [
    # === Bộ luật & Luật cốt lõi (top 10) ===
    "91/2015/QH13",   # Bộ luật Dân sự
    "100/2015/QH13",  # Bộ luật Hình sự
    "101/2015/QH13",  # Bộ luật Tố tụng hình sự
    "45/2019/QH14",   # Bộ luật Lao động
    "31/2024/QH15",   # Luật Đất đai
    "59/2020/QH14",   # Luật Doanh nghiệp
    "36/2024/QH15",   # Luật Trật tự ATGT đường bộ
    "52/2014/QH13",   # Luật Hôn nhân và Gia đình
    "108/2025/QH15",  # Luật Quản lý thuế
    "143/2025/QH15",  # Luật Đầu tư
    # === 5 luật phổ biến đời thường (mở rộng top 15) ===
    "41/2024/QH15",   # Luật Bảo hiểm xã hội
    "27/2023/QH15",   # Luật Nhà ở
    "15/2023/QH15",   # Luật Khám bệnh, chữa bệnh
    "73/2021/QH14",   # Luật Phòng chống ma túy
    "19/2023/QH15",   # Luật Bảo vệ quyền lợi NTD
]


# Throttle theo provider:
#  - Gemini Flash free: 15 RPM → delay 4.1s
#  - Groq llama-3.3-70b free: 30 RPM + 12K TPM → ~6 RPM, delay 10s
#  - Groq llama-3.1-8b free: 30 RPM + 20K TPM → ~8 RPM, delay 7.5s
#  - 9Router (qua Claude Code/GitHub Copilot): cẩn thận rate limit của provider con
DELAYS = {
    "gemini":  4.1,
    "groq":    10.0,
    "groq-8b": 7.5,
    "router9": 10.0,  # 6 RPM — 4s từng gây 429 (mỗi call ~2K token system-prompt của 9Router)
    "openrouter": 4.0,  # GPT (gpt-4o-mini) qua OpenRouter — free-tier ~15-20 RPM
    "kieai": 5.0,       # KieAI deepseek-chat — 3s từng bị rate-limit (choices=None ~60%)
}


def find_raw_file(doc_number: str, raw_dir: Path) -> Optional[tuple[Path, "DocumentMetadata"]]:
    """Tìm file raw có doc_number khớp."""
    for path, meta in iter_raw_files(raw_dir):
        if meta.doc_number == doc_number:
            return path, meta
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1: Extract semantic KG (Offense/Penalty/Subject)")
    p.add_argument("--laws", default=None,
                   help="Comma-separated list of doc_number (default: top 10)")
    p.add_argument("--limit-articles", type=int, default=None,
                   help="Chỉ extract N điều đầu mỗi luật (test)")
    p.add_argument("--dry-run", action="store_true",
                   help="Chỉ in extraction, không write Neo4j")
    p.add_argument("--save-raw", default=None,
                   help="Path file JSON để save raw extraction (debug)")
    p.add_argument("--provider", default="gemini,groq",
                   help="LLM provider chain, comma-separated. Auto-switch khi provider trước hết quota. "
                        "(default: 'gemini,groq')")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip Article đã có ít nhất 1 edge PENALIZES/IMPOSES/APPLIES_TO")
    p.add_argument("--delay", type=float, default=None,
                   help="Override throttle giây/điều (mặc định theo provider trong DELAYS).")
    args = p.parse_args()

    laws_list = (
        [s.strip() for s in args.laws.split(",")] if args.laws
        else DEFAULT_TOP_LAWS
    )
    # Mỗi entry: "groq" (provider mặc định model) HOẶC "router9:ag/gemini-3-flash-agent"
    # (provider:model — xoay vòng nhiều model 9Router). Split ":" đầu tiên nếu vế trái là
    # provider hợp lệ; model name có thể chứa ":" (vd ollama/gpt-oss:120b) nên chỉ split 1 lần.
    def _split_entry(entry: str) -> tuple[str, Optional[str]]:
        if ":" in entry:
            base, model = entry.split(":", 1)
            if base in DELAYS:
                return base, model
        return entry, None

    providers_chain = [p.strip() for p in args.provider.split(",") if p.strip()]
    chain = [_split_entry(e) for e in providers_chain]
    for base, _model in chain:
        if base not in DELAYS:
            print(f"⚠ Unknown provider '{base}', valid: {list(DELAYS.keys())}")
            sys.exit(1)

    print(f"Phase 1 — Semantic KG extraction")
    print(f"  Provider chain: {' → '.join(providers_chain)} (auto-fallback khi quota hết)")
    print(f"  Laws to process: {len(laws_list)}")
    print(f"  Limit articles/law: {args.limit_articles or 'ALL'}")
    print(f"  Skip existing: {args.skip_existing}")
    print(f"  Dry-run: {args.dry_run}\n")

    # ── Init multi-provider extractors (lazy) ──────────────────────────────
    extractors_cache: dict[str, SemanticExtractor] = {}
    current_provider_idx = 0

    def get_extractor():
        """Lấy extractor hiện tại theo current_provider_idx (key = full entry string)."""
        key = providers_chain[current_provider_idx]
        base, model = chain[current_provider_idx]
        if key not in extractors_cache:
            extractors_cache[key] = SemanticExtractor(provider=base, model=model)
            print(f"  ⚡ Khởi tạo provider: {key} (model: {extractors_cache[key].model})")
        return extractors_cache[key], key, (args.delay if args.delay else DELAYS[base])

    # Khởi tạo provider đầu tiên ngay để fail-fast nếu API key sai
    _ = get_extractor()

    client: Optional[Neo4jClient] = None if args.dry_run else Neo4jClient.from_env()

    if client:
        client.create_constraints()

    # Load set of Article IDs đã có extraction (cho --skip-existing)
    existing_articles: set[str] = set()
    if args.skip_existing and client:
        print("Đang đọc Article đã extract (có edge semantic)...")
        with client.session() as s:
            rows = s.run("""
                MATCH (a:Article)
                WHERE a.semantic_done IS NOT NULL
                RETURN a.id AS id
            """).data()
            existing_articles = {r["id"] for r in rows}
        print(f"  → {len(existing_articles)} Article đã có semantic, sẽ skip.\n")

    raw_extractions = []  # for --save-raw

    n_articles_total = 0
    n_articles_ok = 0
    n_articles_err = 0
    n_offenses = 0
    n_penalties = 0
    n_subjects = 0
    t_start = time.time()
    all_quota_exhausted = False

    # ── Loop từng luật ──────────────────────────────────────────────────────
    for law_idx, doc_number in enumerate(laws_list, 1):
        if all_quota_exhausted:
            break
        print(f"\n[{law_idx}/{len(laws_list)}] Luật {doc_number}")

        found = find_raw_file(doc_number, config.RAW_DIR)
        if found is None:
            print(f"  ⚠ Không tìm thấy file raw cho {doc_number}, skip.")
            continue
        path, meta = found
        print(f"  File: {path.parent.name}/{path.name}")

        # Parse text → iterate articles
        try:
            doc = load_document(path, meta)
        except Exception as exc:
            print(f"  ⚠ Lỗi parse: {exc}")
            continue

        normalized = _normalize_for_kg(doc.text)
        # law_id theo cùng convention của structural_extractor
        from src.kg.structural_extractor import _law_id_from_source
        law_id = _law_id_from_source(meta.source)

        articles = list(_iter_articles(normalized))
        articles = [(c, a, t) for c, a, t in articles if a is not None]
        if args.limit_articles:
            articles = articles[: args.limit_articles]
        print(f"  → {len(articles)} điều sẽ extract")

        n_skipped_in_law = 0
        for i, (_chap, art_label, art_text) in enumerate(articles, 1):
            art_id = _article_id(law_id, art_label)

            if args.skip_existing and art_id in existing_articles:
                n_skipped_in_law += 1
                continue

            n_articles_total += 1

            # Lấy extractor hiện tại (có thể đã switch ở vòng trước)
            extractor, current_provider, delay_sec = get_extractor()

            t0 = time.time()
            result = extractor.extract(art_id, art_text)
            elapsed = time.time() - t0

            # Quota exhausted? → switch sang provider tiếp theo nếu có
            if result.quota_exhausted:
                if current_provider_idx + 1 < len(providers_chain):
                    current_provider_idx += 1
                    next_p = providers_chain[current_provider_idx]
                    print(f"    ⚡ Quota {current_provider} hết → switch sang {next_p}, retry điều này")
                    # Retry với provider mới
                    extractor, current_provider, delay_sec = get_extractor()
                    t0 = time.time()
                    result = extractor.extract(art_id, art_text)
                    elapsed = time.time() - t0
                else:
                    print(f"    ❌ Tất cả {len(providers_chain)} providers đã hết quota. STOP.")
                    n_articles_err += 1
                    all_quota_exhausted = True
                    break

            if result.error:
                n_articles_err += 1
                print(f"    [{i:3d}] Điều {art_label}: ❌ {result.error[:80]}")
            else:
                n_articles_ok += 1
                n_offenses += len(result.offenses)
                n_penalties += len(result.penalties)
                n_subjects += len(result.subjects)
                marker = "·" if result.is_empty else "★"
                print(
                    f"    [{i:3d}] Điều {art_label}: {marker} "
                    f"O={len(result.offenses)} P={len(result.penalties)} "
                    f"S={len(result.subjects)} ({elapsed:.1f}s)"
                )

                if not args.dry_run and not result.is_empty:
                    try:
                        client.write_semantic_extraction(
                            article_id=art_id,
                            offenses=result.offenses,
                            penalties=result.penalties,
                            subjects=result.subjects,
                            relations=result.relations,
                        )
                    except Exception as exc:
                        print(f"          ⚠ Lỗi write Neo4j: {exc}")

                # Đánh dấu đã xử lý (kể cả rỗng) để resume bỏ qua, tránh làm lại
                if not args.dry_run:
                    try:
                        client.mark_article_done(art_id)
                    except Exception:
                        pass

            if args.save_raw:
                raw_extractions.append({
                    "article_id": art_id,
                    "law": doc_number,
                    "article": art_label,
                    "offenses": result.offenses,
                    "penalties": result.penalties,
                    "subjects": result.subjects,
                    "relations": result.relations,
                    "error": result.error,
                })

            # Throttle theo provider
            sleep_needed = delay_sec - elapsed
            if sleep_needed > 0:
                time.sleep(sleep_needed)

        if args.skip_existing and n_skipped_in_law:
            print(f"    (đã skip {n_skipped_in_law} điều có sẵn semantic)")

    # ── Summary ─────────────────────────────────────────────────────────────
    elapsed_total = time.time() - t_start
    print("\n" + "=" * 60)
    print(f"Total: {n_articles_total} điều ({n_articles_ok} OK, {n_articles_err} lỗi)")
    print(f"Extracted: {n_offenses} Offenses, {n_penalties} Penalties, {n_subjects} Subjects")
    print(f"Time: {elapsed_total/60:.1f} phút ({elapsed_total/max(n_articles_total,1):.2f}s/điều avg)")

    if client:
        stats = client.stats()
        print("\nStats Neo4j:")
        for label, n in stats["nodes"].items():
            print(f"  ({label}): {n}")
        for rtype, n in stats["relations"].items():
            print(f"  -[{rtype}]->: {n}")
        client.close()

    if args.save_raw:
        Path(args.save_raw).write_text(
            json.dumps(raw_extractions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"\nRaw extractions saved: {args.save_raw}")


if __name__ == "__main__":
    main()
