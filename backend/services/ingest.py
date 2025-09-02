from typing import List, Dict, Any

from ..adapters.pdf import extract_pages_from_pdf_bytes
from ..adapters.docling import parse_to_docling, docling_to_chunks, save_docling_json
from ..core.config import get_settings
from ..models.chunk import Chunk


def simple_chunk_pages(pages: List[tuple[int, str]], max_chars: int = 2000, overlap: int = 200) -> List[Chunk]:
    chunks: List[Chunk] = []
    for page_num, text in pages:
        start = 0
        n = len(text)
        while start < n:
            end = min(n, start + max_chars)
            chunk_text = text[start:end]
            chunk_id = f"p{page_num}_{start}"
            chunks.append(Chunk(chunk_id=chunk_id, page=page_num, text=chunk_text))
            if end == n:
                break
            start = end - overlap
            if start < 0:
                start = 0
    return chunks


def ingest_pdf_bytes(data: bytes) -> List[Chunk]:
    """
    Prefer Docling for rich, structure-aware chunking. Fallback to pypdf windowing.
    """
    # Try Docling first
    try:
        doc = parse_to_docling(data)
        chunks = docling_to_chunks(doc)
        # Save Docling JSON cache if configured
        try:
            settings = get_settings()
            cache_dir = getattr(settings, "data_root", "backend/data") + "/docling"
            save_docling_json(cache_dir, "last_ingest", doc)
        except Exception:
            pass
        return chunks
    except Exception:
        # Fallback: simple page chunking
        pages = extract_pages_from_pdf_bytes(data)
        return simple_chunk_pages(pages)
