from typing import List, Tuple


def extract_pages_from_pdf_bytes(data: bytes) -> List[Tuple[int, str]]:
    """
    Returns a list of (page_number, text) tuples.
    Uses pypdf if available; otherwise, returns a placeholder.
    """
    try:
        import io
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        pages: List[Tuple[int, str]] = []
        for i, p in enumerate(reader.pages):
            text = p.extract_text() or ""
            pages.append((i + 1, text))
        return pages
    except Exception:
        # Fallback placeholder to keep the pipeline non-breaking during scaffold
        return [(1, "")]  # Signal minimal content

