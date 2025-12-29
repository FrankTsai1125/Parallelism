import argparse
from pathlib import Path
import sys


def _parse_pages(s: str, total: int) -> list[int]:
    # Accept formats like: "1-3", "1,2,5", "all"
    s = s.strip().lower()
    if s in {"all", "*"}:
        return list(range(total))

    pages: set[int] = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a_i = int(a)
            b_i = int(b)
            lo = min(a_i, b_i)
            hi = max(a_i, b_i)
            for p in range(lo, hi + 1):
                pages.add(p - 1)  # 1-based -> 0-based
        else:
            pages.add(int(part) - 1)

    return [p for p in sorted(pages) if 0 <= p < total]


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract text from a PDF (basic).")
    ap.add_argument("pdf_path", type=Path)
    ap.add_argument("--pages", default="1-2", help='Pages to extract (1-based). Ex: "1-2", "1,3,5", "all"')
    ap.add_argument("--max-chars", type=int, default=12000, help="Max chars per page to print")
    args = ap.parse_args()

    # Windows consoles may default to cp950/cp1252, which can crash on extracted Unicode text.
    # Force UTF-8 and replace non-encodable chars to keep extraction running.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    from pypdf import PdfReader

    reader = PdfReader(str(args.pdf_path))
    total = len(reader.pages)
    pages = _parse_pages(args.pages, total)

    print(f"pages_total={total}")
    print(f"pages_selected={','.join(str(p+1) for p in pages)}")

    for idx in pages:
        text = reader.pages[idx].extract_text() or ""
        text = "\n".join(line.rstrip() for line in text.splitlines())
        print(f"\n--- PAGE {idx+1} ---\n")
        print(text[: args.max_chars])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


