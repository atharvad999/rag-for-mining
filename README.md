# RTE Backend (Scaffold)

Retrieval‑augmented generation backend for mining documents: ingest PDFs, build a global knowledge base, and expose semantic search, summarization, and Q&A via FastAPI.

## Quick start

1) Ensure `.env` is populated (Supabase + Groq + embeddings). See plan.md.
2) Create venv and install deps:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
3) Run the API:
```
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```
4) Health check: GET http://localhost:8000/health

## Knowledge Base (docs/ → global index)
- Build a global KB index from all PDFs in `docs/`:
```
source venv2/bin/activate
python scripts/build_kb_index.py --dir docs --kb-id kb_global
```
- Ask questions against the KB via the existing endpoint (use `kb_global` as the tender_id):
```
curl -X POST http://localhost:8000/tenders/kb_global/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the EMD across these tenders?"}'
```
The response citations include `section_hint` with the source filename where applicable.

## Docling Ingestion (structured parsing)
- Install: ensure `docling` and OCR system deps are present (Tesseract/Poppler as per Docling docs).
- Behavior: ingestion prefers Docling for structured parsing (sections, paragraphs, tables). Falls back to `pypdf` if Docling is unavailable or parsing fails.
- Outputs: chunks include `section_hint` with a semantic path (e.g., `filename | Section > Subsection > para`), improving citation quality.
- Rebuild KB after adding new PDFs: re-run `scripts/build_kb_index.py` to refresh FAISS artifacts.

## Structure
- `backend/app.py` FastAPI app with health, upload, summary, Q&A (placeholders)
- `backend/core/config.py` Settings loader from env with validation
- `backend/models/schema.py` Pydantic models
- `backend/adapters/*` Supabase, Groq, PDF, Embeddings (stubs)
- `backend/services/*` ingest, retriever, summary, qa (stubs)
- `scripts/ingest_file.py` CLI to parse a PDF into chunks

## Next steps
- Wire Supabase client in `adapters/storage_supabase.py` and persist tender metadata.
- Implement FAISS retriever and replace `SimpleRetriever`.
- Implement Groq LLM calls and OpenAI embeddings.
- Flesh out summary extraction rules and Q&A prompts.
