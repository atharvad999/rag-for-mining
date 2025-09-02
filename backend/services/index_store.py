import json
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np

from ..models.chunk import Chunk


def _safe_id(tender_id: str) -> str:
    return tender_id.replace("/", "_").replace("\\", "_")


def save_index(index_root: str, tender_id: str, chunks: List[Chunk], embeddings: List[List[float]]):
    import faiss  # type: ignore

    os.makedirs(index_root, exist_ok=True)
    sid = _safe_id(tender_id)
    base = Path(index_root) / sid
    base.mkdir(parents=True, exist_ok=True)

    # Save chunks metadata
    chunks_path = base / "chunks.json"
    with chunks_path.open("w", encoding="utf-8") as f:
        json.dump([
            {"chunk_id": c.chunk_id, "page": c.page, "text": c.text, "section_hint": c.section_hint}
            for c in chunks
        ], f)

    # Build and save FAISS index
    vecs = np.array(embeddings, dtype="float32")
    dim = vecs.shape[1]
    index = faiss.IndexFlatIP(dim)
    # Normalize for cosine similarity
    faiss.normalize_L2(vecs)
    index.add(vecs)
    faiss.write_index(index, str(base / "index.faiss"))


def load_index(index_root: str, tender_id: str) -> Tuple[List[Chunk], "faiss.Index"]:
    import faiss  # type: ignore

    sid = _safe_id(tender_id)
    base = Path(index_root) / sid
    chunks_path = base / "chunks.json"
    index_path = base / "index.faiss"

    if not chunks_path.exists() or not index_path.exists():
        raise FileNotFoundError("Index not found. Run ingestion first.")

    raw = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [Chunk(chunk_id=r["chunk_id"], page=r["page"], text=r["text"], section_hint=r.get("section_hint")) for r in raw]
    index = faiss.read_index(str(index_path))
    return chunks, index
