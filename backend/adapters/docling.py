from __future__ import annotations

"""
Docling adapter: parse PDFs into a structured document and convert to Chunk list.

This module is defensive: it attempts multiple import paths and gracefully falls back
if Docling is not available or the structure is unexpected. When Docling succeeds,
we preserve rich metadata (file, page, section path) into Chunk.section_hint.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json

from ..models.chunk import Chunk


def _try_import_docling():
    """Attempt to import Docling APIs in a few common layouts."""
    # Layout 1: documented converter API
    try:
        from docling.document_converter import DocumentConverter  # type: ignore
        return {"style": 1, "DocumentConverter": DocumentConverter}
    except Exception:
        pass
    # Layout 2: pipeline/processor style
    try:
        from docling import DocumentConverter as DC  # type: ignore
        return {"style": 2, "DocumentConverter": DC}
    except Exception:
        pass
    # Layout 3: expose only parse util
    try:
        from docling import parse as dl_parse  # type: ignore
        return {"style": 3, "parse": dl_parse}
    except Exception:
        pass
    return None


def parse_to_docling(pdf_bytes: bytes) -> Any:
    """
    Returns a Docling document object if possible, else raises.
    """
    dl = _try_import_docling()
    if not dl:
        raise RuntimeError("Docling is not available in this environment")

    style = dl.get("style")
    if style in (1, 2):
        DC = dl["DocumentConverter"]
        # Try multiple strategies for different Docling versions
        last_err: Optional[Exception] = None
        # Strategy A: convert_bytes
        try:
            conv = DC()
            doc = conv.convert_bytes(pdf_bytes, mime_type="application/pdf")  # type: ignore
            return doc
        except Exception as e:
            last_err = e
        # Strategy B: convert(file-like)
        try:
            import io
            conv = DC()
            doc = conv.convert(io.BytesIO(pdf_bytes))  # type: ignore
            return doc
        except Exception as e:
            last_err = e
        # Strategy C: write to temp file and convert(path)
        try:
            import tempfile, os
            from pathlib import Path
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "input.pdf"
                p.write_bytes(pdf_bytes)
                conv = DC()
                doc = conv.convert(str(p))  # type: ignore
                return doc
        except Exception as e:
            last_err = e
        raise RuntimeError(f"Docling conversion failed: {last_err}")
    elif style == 3:
        try:
            doc = dl["parse"](pdf_bytes)
            return doc
        except Exception as e:
            raise RuntimeError(f"Docling parse failed: {e}")
    else:
        raise RuntimeError("Unknown Docling import style")


def _docling_to_json(doc: Any) -> Optional[Dict[str, Any]]:
    """Try to obtain a JSON-like dict from a Docling document."""
    try:
        if hasattr(doc, "to_json"):
            raw = doc.to_json()  # type: ignore
            if isinstance(raw, str):
                return json.loads(raw)
            if isinstance(raw, dict):
                return raw
    except Exception:
        pass
    try:
        if hasattr(doc, "model_dump"):
            return doc.model_dump()  # pydantic-like
    except Exception:
        pass
    try:
        # last resort: JSON-serialize via __dict__
        return json.loads(json.dumps(doc, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return None


def _iter_text_nodes(obj: Any, section_stack: Optional[List[str]] = None) -> Iterable[Tuple[str, Optional[int], Optional[str]]]:
    """
    Generic walker over a JSON-like Docling structure.
    Yields (text, page, section_path).
    Heuristics: look for keys like 'text', 'page', 'type', 'title', 'heading'.
    """
    if section_stack is None:
        section_stack = []

    if isinstance(obj, dict):
        # Section-ish nodes
        title = None
        for k in ("title", "heading", "name"):
            if k in obj and isinstance(obj[k], str) and obj[k].strip():
                title = obj[k].strip()
                break
        if title:
            section_stack.append(title)

        # Paragraph/text nodes
        if "text" in obj and isinstance(obj["text"], str) and obj["text"].strip():
            text = obj["text"].strip()
            page = None
            for pk in ("page", "page_no", "page_index", "pageNumber"):
                if pk in obj and isinstance(obj[pk], int):
                    page = obj[pk]
                    break
            path = " > ".join(section_stack) if section_stack else None
            yield (text, page, path)

        # Table-like nodes (serialize to simple Markdown-ish text if present)
        if obj.get("type") == "table" and "cells" in obj:
            try:
                rows = obj["cells"]
                lines: List[str] = []
                for r in rows:
                    if isinstance(r, list):
                        lines.append(" | ".join(str(c or "").strip() for c in r))
                if lines:
                    page = obj.get("page") if isinstance(obj.get("page"), int) else None
                    path = " > ".join(section_stack) if section_stack else "Table"
                    yield ("\n".join(lines), page, path)
            except Exception:
                pass

        # Figure/image captions
        if obj.get("type") in {"figure", "image"}:
            cap = None
            for ck in ("caption", "alt", "title"):
                v = obj.get(ck)
                if isinstance(v, str) and v.strip():
                    cap = v.strip()
                    break
            if cap:
                page = obj.get("page") if isinstance(obj.get("page"), int) else None
                path = " > ".join(section_stack) if section_stack else "Figure"
                yield (cap, page, path)

        # Recurse
        for v in obj.values():
            yield from _iter_text_nodes(v, section_stack)

        # Pop section if we pushed one
        if title and section_stack:
            section_stack.pop()

    elif isinstance(obj, list):
        for it in obj:
            yield from _iter_text_nodes(it, section_stack)


def docling_to_chunks(doc: Any, max_chars: int = 2500, overlap: int = 200) -> List[Chunk]:
    """
    Convert Docling document to Chunk list, preserving page and section path.
    Small paragraphs are merged up to ~max_chars with overlap.
    """
    raw = _docling_to_json(doc)
    if raw is None:
        # As a last resort, treat the whole doc as a single chunk of unknown page
        return [Chunk(chunk_id="dl_0", page=1, text=str(doc), section_hint=None)]

    # Collect spans
    spans: List[Tuple[str, Optional[int], Optional[str]]] = []
    for text, page, path in _iter_text_nodes(raw):
        if text:
            spans.append((text, page, path))

    # Merge spans into chunks
    chunks: List[Chunk] = []
    buf: List[str] = []
    buf_pages: List[Optional[int]] = []
    buf_paths: List[Optional[str]] = []

    def flush(start_idx: int, page: Optional[int], path: Optional[str]):
        if not buf:
            return
        text = "\n\n".join(buf).strip()
        if not text:
            return
        cid = f"dl_{start_idx}_{len(chunks)}"
        chunks.append(Chunk(chunk_id=cid, page=page or 1, text=text, section_hint=path))

    start = 0
    cur_len = 0
    current_path: Optional[str] = None
    current_page: Optional[int] = None

    for i, (t, p, path) in enumerate(spans):
        if not buf:
            start = i
            current_path = path
            current_page = p
        # If path changes, flush to keep semantic grouping
        if path != current_path and buf:
            flush(start, current_page, current_path)
            buf, buf_pages, buf_paths = [], [], []
            start = i
            cur_len = 0
            current_path = path
            current_page = p

        buf.append(t)
        buf_pages.append(p)
        buf_paths.append(path)
        cur_len += len(t)
        if cur_len >= max_chars:
            flush(start, current_page, current_path)
            if overlap > 0 and buf:
                # keep tail overlap
                joined = "\n\n".join(buf)
                tail = joined[-overlap:]
                buf = [tail]
                cur_len = len(tail)
                start = i
            else:
                buf = []
                cur_len = 0
                start = i
            current_path = path
            current_page = p

    flush(start, current_page, current_path)
    return chunks


def save_docling_json(cache_dir: str, key: str, doc: Any) -> Optional[str]:
    """Save a JSON version of the Docling doc for caching/debug. Returns file path."""
    import os
    from pathlib import Path

    raw = _docling_to_json(doc)
    if raw is None:
        return None
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    path = os.path.join(cache_dir, f"{key}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(raw, f)
    return path
