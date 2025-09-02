from pydantic import BaseModel, Field
from typing import List, Optional


class HealthStatus(BaseModel):
    status: str
    llm_provider: str
    storage_backend: str


class Citation(BaseModel):
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    section_hint: Optional[str] = None
    text_snippet: Optional[str] = None


class SummaryField(BaseModel):
    value: Optional[str] = None
    confidence: Optional[float] = None
    citations: List[Citation] = Field(default_factory=list)


class SummarySheet(BaseModel):
    tender_name: Optional[str] = None
    issuer: Optional[str] = None
    emd_amount: Optional[str] = None
    location: Optional[str] = None
    duration: Optional[str] = None
    scope_of_work: Optional[str] = None
    compliance_notes: List[str] = Field(default_factory=list)
    citations: List[Citation] = Field(default_factory=list)


class UploadResponse(BaseModel):
    tender_id: str
    filename: str


class QARequest(BaseModel):
    question: str


class QAResponse(BaseModel):
    answer: str
    citations: List[Citation] = Field(default_factory=list)


class IngestResponse(BaseModel):
    tender_id: str
    pages: int
    chunks: int
