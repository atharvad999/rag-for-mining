#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys, os

# Ensure project root is on sys.path when running as a script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.services.ingest import ingest_pdf_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=str, help="Path to PDF file")
    args = ap.parse_args()

    data = Path(args.pdf).read_bytes()
    chunks = ingest_pdf_bytes(data)
    print(f"Parsed {len(chunks)} chunks")
    if chunks:
        c0 = chunks[0]
        hint = f" | {c0.section_hint}" if getattr(c0, 'section_hint', None) else ""
        print(f"Sample: {c0.chunk_id} page={c0.page}{hint} text_len={len(c0.text)}")


if __name__ == "__main__":
    main()
