#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List
import sys, os

# Ensure project root is on sys.path when running as a script
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from backend.core.config import get_settings
from backend.services.ingest import ingest_pdf_bytes
from backend.models.chunk import Chunk
from backend.adapters.embeddings import embed_texts
from backend.services.index_store import save_index


def batched(seq: List[str], size: int):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def main():
    ap = argparse.ArgumentParser(description="Build a global KB FAISS index from a folder of PDFs.")
    ap.add_argument("--dir", default="docs", help="Folder containing PDFs (default: docs)")
    ap.add_argument("--kb-id", default="kb_global", help="ID to save the index under (default: kb_global)")
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size (default: 64)")
    args = ap.parse_args()

    settings = get_settings()

    root = Path(args.dir)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    pdfs = sorted([p for p in root.glob("**/*.pdf") if p.is_file()])
    if not pdfs:
        raise SystemExit(f"No PDFs found under {root}")

    all_chunks: List[Chunk] = []
    for pdf in pdfs:
        data = pdf.read_bytes()
        chunks = ingest_pdf_bytes(data)
        # Preserve Docling-provided section_hint; prefix filename for provenance
        fname = pdf.name
        annotated: List[Chunk] = []
        for c in chunks:
            hint = c.section_hint
            if hint:
                hint = f"{fname} | {hint}"
            else:
                hint = fname
            annotated.append(Chunk(chunk_id=c.chunk_id, page=c.page, text=c.text, section_hint=hint))
        all_chunks.extend(annotated)
        print(f"Parsed {len(chunks):4d} chunks from {fname}")

    texts = [c.text for c in all_chunks]

    print(f"Embedding {len(texts)} chunks using {settings.emb_provider}:{settings.emb_model} ...")
    embeddings: List[List[float]] = []
    for batch in batched(texts, args.batch):
        vecs = embed_texts(batch, provider=settings.emb_provider, model=settings.emb_model, openai_api_key=settings.openai_api_key)
        embeddings.extend(vecs)

    if len(embeddings) != len(all_chunks):
        raise SystemExit("Embedding count mismatch; aborting.")

    print(f"Saving index to {settings.index_root}/{args.kb_id} ...")
    save_index(settings.index_root, args.kb_id, all_chunks, embeddings)
    pages = len({c.page for c in all_chunks})
    print(f"Done. KB ID: {args.kb_id} | pages~{pages} | chunks={len(all_chunks)}")


if __name__ == "__main__":
    main()
