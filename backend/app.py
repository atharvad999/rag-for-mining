from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from .core.config import get_settings, Settings
from .models.schema import HealthStatus, SummarySheet, QARequest, QAResponse, UploadResponse, IngestResponse
from .adapters.storage_supabase import SupabaseStorage
from .adapters.storage_local import LocalStorage
from uuid import uuid4
from .services.ingest import ingest_pdf_bytes
from .services.index_store import save_index, load_index
from .services.retriever import FaissRetriever
from .adapters.embeddings import embed_texts
from .adapters.llm_groq import GroqLLM
from .models.schema import Citation
from .services.summarizer import extract_summary_groq

app = FastAPI(title="RTE Backend", version="0.1.0")


@app.on_event("startup")
async def on_startup():
    settings = get_settings()
    # In a fuller implementation, initialize shared clients here.
    # e.g., Supabase, DB pool, embedding model, LLM client.
    return settings


# CORS
settings = get_settings()
origins = ["*"] if settings.cors_origins == "*" else [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


def get_settings_dep() -> Settings:
    return get_settings()


@app.get("/health", response_model=HealthStatus)
async def health(settings: Settings = Depends(get_settings_dep)):
    return HealthStatus(status="ok", llm_provider=settings.llm_provider, storage_backend=settings.storage_backend)


@app.post("/upload", response_model=UploadResponse)
async def upload_tender(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings_dep),
):
    # Placeholder: Ingest file into Supabase Storage and return an ID.
    if file.content_type not in {"application/pdf"}:
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    data = await file.read()
    object_path = f"tenders/{uuid4()}_{file.filename}"
    try:
        if settings.storage_backend == "local":
            storage_root = f"{settings.data_root}/storage"
            LocalStorage(storage_root).upload_file(object_path, data, content_type="application/pdf")
        else:
            storage = SupabaseStorage(
                url=settings.supabase_url or "",
                service_role_key=settings.supabase_service_role_key or "",
                bucket=settings.supabase_storage_bucket,
            )
            storage.upload_file(object_path, data, content_type="application/pdf")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    tender_id = object_path
    return UploadResponse(tender_id=tender_id, filename=file.filename)


@app.get("/tenders/{tender_id}/summary", response_model=SummarySheet)
async def get_summary(tender_id: str):
    # Normalize ID for local storage if caller passed just a filename
    try:
        settings = get_settings()
        if settings.storage_backend == "local" and "/" not in tender_id:
            tender_id = f"tenders/{tender_id}"
    except Exception:
        pass
    # Try to load precomputed summary; else compute if possible
    from pathlib import Path
    import json
    from .services.index_store import _safe_id  # type: ignore
    sid = _safe_id(tender_id)
    base = Path(get_settings().index_root) / sid
    sp = base / "summary.json"
    if sp.exists():
        try:
            data = json.loads(sp.read_text(encoding="utf-8"))
            def _empty(d: dict) -> bool:
                return not any(d.get(k) for k in ["tender_name", "issuer", "emd_amount", "location", "duration", "scope_of_work"]) and not (d.get("compliance_notes") or [])
            if isinstance(data, dict) and not _empty(data):
                return SummarySheet(
                    tender_name=data.get("tender_name") or tender_id,
                    issuer=data.get("issuer"),
                    emd_amount=data.get("emd_amount"),
                    location=data.get("location"),
                    duration=data.get("duration"),
                    scope_of_work=data.get("scope_of_work"),
                    compliance_notes=data.get("compliance_notes") or [],
                    citations=[],
                )
            # else fall through to recompute below
        except Exception:
            pass

    try:
        chunks, _ = load_index(get_settings().index_root, tender_id)
        settings = get_settings()
        if settings.groq_api_key:
            summary, _cites = extract_summary_groq(chunks, api_key=settings.groq_api_key, model=settings.llm_model)
            cites = [
                Citation(page=c.page, chunk_id=c.chunk_id, section_hint=c.section_hint, text_snippet=c.text[:160])
                for c in _cites
            ]
            return SummarySheet(
                tender_name=summary.get("tender_name") or tender_id,
                issuer=summary.get("issuer"),
                emd_amount=summary.get("emd_amount"),
                location=summary.get("location"),
                duration=summary.get("duration"),
                scope_of_work=summary.get("scope_of_work"),
                compliance_notes=summary.get("compliance_notes") or [],
                citations=cites,
            )
    except Exception:
        pass

    return SummarySheet(tender_name=tender_id, compliance_notes=[], citations=[])


@app.post("/tenders/{tender_id}/ask", response_model=QAResponse)
async def ask_question(tender_id: str, req: QARequest):
    # Use saved FAISS index + chunks to answer
    try:
        chunks, index = load_index(get_settings().index_root, tender_id)
    except FileNotFoundError:
        return QAResponse(answer="Index not found. Please ingest first.", citations=[])

    settings = get_settings()
    retriever = FaissRetriever(
        chunks=chunks,
        index=index,
        emb_provider=settings.emb_provider,
        emb_model=settings.emb_model,
        openai_api_key=settings.openai_api_key,
    )
    top = retriever.query(req.question, top_k=settings.top_k)

    # Build prompt with context only (omit headers that models may echo)
    context_blocks = []
    for c, _score in top:
        context_blocks.append(c.text)
    context = "\n\n---\n\n".join(context_blocks)
    prompt = (
        "You are a helpful tender Q&A assistant.\n"
        "- Answer strictly from the context. If the answer is not present, reply exactly: Not found in tender.\n"
        "- Return only the answer text. Do not include filenames, IDs, or any preamble.\n"
        "- Be concise.\n\n"
        f"Context:\n{context}\n\nQuestion: {req.question}\nAnswer:"
    )

    llm = GroqLLM(api_key=settings.groq_api_key or "", model=settings.llm_model)
    import re
    answer = llm.complete(prompt, temperature=0.0, max_tokens=256)
    # Post-process to avoid filename/id echoes
    ans = (answer or "").strip()
    # Remove leading label
    ans = re.sub(r"^Answer\s*:\s*", "", ans, flags=re.IGNORECASE).strip()
    # Collapse whitespace
    ans = re.sub(r"\s+", " ", ans)
    if not ans or ans == tender_id or re.search(r"\.pdf\b", ans, flags=re.IGNORECASE):
        ans = "Not found in tender"

    cites = [
        Citation(page=c.page, chunk_id=c.chunk_id, section_hint=c.section_hint, text_snippet=c.text[:160])
        for c, _ in top
    ]
    return QAResponse(answer=ans, citations=cites)


@app.post("/tenders/{tender_id}/ingest", response_model=IngestResponse)
async def ingest_tender(tender_id: str, settings: Settings = Depends(get_settings_dep)):
    # tender_id is the storage path returned from upload (e.g., tenders/<uuid>_file.pdf)
    # Normalize ID for local storage if caller passed just a filename
    if settings.storage_backend == "local" and "/" not in tender_id:
        tender_id = f"tenders/{tender_id}"
    # Download bytes from selected backend
    try:
        if settings.storage_backend == "local":
            storage_root = f"{settings.data_root}/storage"
            data = LocalStorage(storage_root).download_file(tender_id)
        else:
            storage = SupabaseStorage(
                url=settings.supabase_url or "",
                service_role_key=settings.supabase_service_role_key or "",
                bucket=settings.supabase_storage_bucket,
            )
            client = storage.client
            if client is None:
                raise RuntimeError("Supabase client not initialized")
            obj = client.storage.from_(settings.supabase_storage_bucket).download(tender_id)
            data = obj  # bytes
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

    # Ingest
    chunks = ingest_pdf_bytes(data)
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts, provider=settings.emb_provider, model=settings.emb_model, openai_api_key=settings.openai_api_key)

    # Save FAISS + chunks
    save_index(settings.index_root, tender_id, chunks, embeddings)

    # Best-effort: precompute and cache summary using Groq if configured
    try:
        if settings.groq_api_key:
            summary, _cites = extract_summary_groq(chunks, api_key=settings.groq_api_key, model=settings.llm_model)
            def _empty(d: dict) -> bool:
                return not any(d.get(k) for k in ["tender_name", "issuer", "emd_amount", "location", "duration", "scope_of_work"]) and not (d.get("compliance_notes") or [])
            if not _empty(summary):
                import json
                from pathlib import Path
                from .services.index_store import _safe_id  # type: ignore
                sid = _safe_id(tender_id)
                base = Path(settings.index_root) / sid
                with open(base / "summary.json", "w", encoding="utf-8") as f:
                    json.dump(summary, f)
    except Exception:
        pass

    # Respond
    pages = len({c.page for c in chunks})
    return IngestResponse(tender_id=tender_id, pages=pages, chunks=len(chunks))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.app_host, port=settings.app_port)
