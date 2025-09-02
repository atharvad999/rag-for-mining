from typing import List, Optional
from ..models.schema import SummarySheet, Citation
from .ingest import Chunk


def extract_summary(chunks: List[Chunk]) -> SummarySheet:
    # Minimal placeholder implementation
    citations: List[Citation] = []
    if chunks:
        citations.append(Citation(page=chunks[0].page, chunk_id=chunks[0].chunk_id, text_snippet=chunks[0].text[:160]))
    return SummarySheet(
        tender_name=None,
        issuer=None,
        emd_amount=None,
        location=None,
        duration=None,
        scope_of_work=None,
        compliance_notes=[],
        citations=citations,
    )

