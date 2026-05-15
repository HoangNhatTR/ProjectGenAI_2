"""Chạy thử parsing + cleaning trên các file trong data/raw/.

Cách chạy (từ thư mục ProjectGenAI_2, sau khi đã activate venv):
    python -m scripts.demo_parsing
"""
from __future__ import annotations

from pathlib import Path

from src import config
from src.parsing import load_document, clean_text, parse_txt, parse_html, parse_pdf
from src.schemas import DocumentMetadata


PARSERS = {".txt": parse_txt, ".html": parse_html, ".htm": parse_html, ".pdf": parse_pdf}


def show(path: Path) -> None:
    print("=" * 70)
    print(f"FILE: {path.name}  ({path.stat().st_size} bytes)")
    print("=" * 70)

    parser = PARSERS.get(path.suffix.lower())
    if parser is None:
        print(f"[skip] không hỗ trợ {path.suffix}")
        return

    raw = parser(path)
    cleaned = clean_text(raw)

    print(f"raw    : {len(raw):>6} chars / {len(raw.splitlines()):>4} lines")
    print(f"cleaned: {len(cleaned):>6} chars / {len(cleaned.splitlines()):>4} lines")
    print()
    print("--- 500 ký tự đầu (cleaned) ---")
    print(cleaned[:500])
    print()
    print("--- 200 ký tự cuối (cleaned) ---")
    print(cleaned[-200:])
    print()

    meta = DocumentMetadata(source=path.name)
    doc = load_document(path, meta)
    print(f"RawDocument OK  | text={len(doc.text)} chars | metadata={doc.metadata.model_dump()}")
    print()


def main() -> None:
    raw_dir = config.RAW_DIR
    if not raw_dir.exists():
        print(f"[!] Không thấy thư mục {raw_dir}")
        return

    files = sorted(p for p in raw_dir.iterdir() if p.is_file())
    if not files:
        print(f"[!] {raw_dir} rỗng, hãy bỏ vài file vào rồi chạy lại")
        return

    for f in files:
        show(f)


if __name__ == "__main__":
    main()
