from typing import List
from ..models.schema import QAResponse, Citation
from .ingest import Chunk


def answer_question(question: str, top_chunks: List[Chunk]) -> QAResponse:
    # Placeholder: always return Not found with citations of provided chunks
    cits = [Citation(page=c.page, chunk_id=c.chunk_id, text_snippet=c.text[:160]) for c in top_chunks]
    return QAResponse(answer="Not found in tender", citations=cits)

