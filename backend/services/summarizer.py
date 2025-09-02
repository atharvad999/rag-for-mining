from __future__ import annotations

import json
from typing import List, Optional, Tuple
import re

from ..models.chunk import Chunk
from ..adapters.llm_groq import GroqLLM


SUMMARY_FIELDS = [
    "tender_name",
    "issuer",
    "emd_amount",
    "location",
    "duration",
    "scope_of_work",
    "compliance_notes",
]


def build_summary_prompt(chunks: List[Chunk], question_hint: Optional[str] = None, max_chars: int = 8000) -> str:
    # Concatenate early chunks/pages as context
    buf: List[str] = []
    total = 0
    for c in chunks:
        span = f"[page {c.page} | {c.chunk_id} | {c.section_hint or ''}]\n{c.text}\n\n"
        if total + len(span) > max_chars:
            break
        buf.append(span)
        total += len(span)
    context = "".join(buf)
    instructions = (
        "You are a tender document analyzer. Extract the following fields as JSON with keys: "
        + ", ".join(SUMMARY_FIELDS)
        + ".\n"
        "- tender_name: Short name or title of the tender.\n"
        "- issuer: The issuing organization.\n"
        "- emd_amount: Earnest Money Deposit value (with currency in ruppees).\n"
        "- location: Primary location(s) of work.\n"
        "- duration: Contract/project duration.\n"
        "- scope_of_work: 1-3 sentence summary of key scope.\n"
        "- compliance_notes: array of 3-8 short bullets for critical compliance/eligibility/financial terms.\n\n"
        "Return ONLY valid JSON. If a value is not found, use null (or [] for arrays)."
    )
    return f"{instructions}\n\nContext:\n{context}"


def extract_summary_groq(chunks: List[Chunk], api_key: str, model: str = "llama3-70b-8192") -> Tuple[dict, List[Chunk]]:
    """Use Groq LLM to extract summary fields from chunks. Returns (summary_dict, cited_chunks)."""
    if not chunks:
        return {k: None for k in SUMMARY_FIELDS}, []

    prompt = build_summary_prompt(chunks)
    llm = GroqLLM(api_key=api_key, model=model)
    raw = llm.complete(prompt, temperature=0.0, max_tokens=600)

    def _extract_json_block(text: str) -> Optional[str]:
        """Try to extract a JSON object from arbitrary LLM text.
        Handles code fences and extra prose. Returns the JSON substring or None.
        """
        if not text:
            return None
        # 1) Look for fenced ```json ... ``` blocks
        m = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
        # 2) Look for first {...} block by counting braces
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i, ch in enumerate(text[start:], start=start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    return text[start:i+1]
        return None

    def _coerce_summary(d: dict) -> dict:
        out = {k: None for k in SUMMARY_FIELDS}
        if not isinstance(d, dict):
            return out
        for k in SUMMARY_FIELDS:
            v = d.get(k)
            if k == "compliance_notes":
                if isinstance(v, list):
                    # Keep only strings, coerce others via str()
                    out[k] = [str(x) for x in v if x is not None]
                else:
                    out[k] = []
            else:
                if v is None:
                    out[k] = None
                else:
                    out[k] = str(v)
        return out

    summary: dict
    json_str = _extract_json_block(raw)
    if json_str:
        try:
            summary = _coerce_summary(json.loads(json_str))
        except Exception:
            summary = {k: None for k in SUMMARY_FIELDS}
    else:
        # Final fallback: try direct parse
        try:
            summary = _coerce_summary(json.loads(raw))
        except Exception:
            summary = {k: None for k in SUMMARY_FIELDS}

    # If everything is empty/null, try a light rule-based extraction as fallback
    def _is_effectively_empty(d: dict) -> bool:
        if not d:
            return True
        non_null = sum(1 for k, v in d.items() if v not in (None, [], ""))
        return non_null == 0

    if _is_effectively_empty(summary):
        rb_sum, rb_cites = extract_summary_rules(chunks)
        return rb_sum, rb_cites

    # Capture first few chunks as coarse "citations"
    cites = chunks[:5]
    return summary, cites


def extract_summary_rules(chunks: List[Chunk]) -> Tuple[dict, List[Chunk]]:
    """Very light heuristic extraction from first few chunks as a safety net."""
    head_chunks = chunks[:5]
    text = "\n".join(c.text for c in head_chunks)

    # Tender name: pick first non-empty, reasonably short header-ish line
    tender_name = None
    for line in text.splitlines():
        s = line.strip()
        if len(s) >= 6 and len(s) <= 140:
            if not re.search(r"\b(page\s*\d+|table of contents)\b", s, flags=re.I):
                tender_name = s
                break

    # Issuer: look for organization-like lines
    issuer = None
    m = re.search(r"\b(?:Corporation|Company|Department|Ministry|Government|Govt\.?|Ltd\.?|Limited|Authority|NMDC|BIOM)[:\s,\-]*([^\n]{3,80})", text, flags=re.I)
    if m:
        issuer = m.group(0).strip()

    # EMD amount
    emd_amount = None
    m = re.search(r"(?:EMD|Earnest Money(?: Deposit)?)[^\n:]*[:\-]?\s*(₹|INR|Rs\.?|RUPEES)?\s*([\d,]+(?:\.\d{1,2})?)", text, flags=re.I)
    if m:
        cur = m.group(1) or ""
        val = m.group(2)
        emd_amount = (cur + " " + val).strip()

    # Duration
    duration = None
    m = re.search(r"(?:Duration|Period)[^\n:]*[:\-]?\s*([\d]+\s*(?:day|month|year)s?)", text, flags=re.I)
    if m:
        duration = m.group(1)

    # Location
    location = None
    m = re.search(r"(?:Location|Place of work)[^\n:]*[:\-]?\s*([^\n]{3,80})", text, flags=re.I)
    if m:
        location = m.group(1).strip()

    # Scope of Work: grab up to ~2 sentences after heading
    scope_of_work = None
    m = re.search(r"Scope of Work[\s\-:]*([\s\S]{0,500})", text, flags=re.I)
    if m:
        snippet = m.group(1).strip()
        # simple sentence cutoff
        parts = re.split(r"(?<=[.!?])\s+", snippet)
        scope_of_work = " ".join(parts[:2]).strip()[:300]

    # Compliance notes: pick lines that look like bullets and contain key terms
    compliance_notes: List[str] = []
    key_terms = [
        r"eligibility", r"turnover", r"experience", r"bid security", r"emd", r"bank guarantee", r"penalty", r"liquidated damages",
    ]
    for line in text.splitlines():
        s = line.strip(" -*•\t")
        if 6 <= len(s) <= 160 and any(re.search(t, s, flags=re.I) for t in key_terms):
            compliance_notes.append(s)
            if len(compliance_notes) >= 6:
                break

    summary = {
        "tender_name": tender_name,
        "issuer": issuer,
        "emd_amount": emd_amount,
        "location": location,
        "duration": duration,
        "scope_of_work": scope_of_work,
        "compliance_notes": compliance_notes,
    }
    cites = head_chunks
    return summary, cites
