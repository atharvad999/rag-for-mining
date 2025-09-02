from typing import List, Optional
import hashlib


def embed_texts(
    texts: List[str],
    provider: str = "openai",
    model: str = "text-embedding-3-small",
    openai_api_key: Optional[str] = None,
) -> List[List[float]]:
    """Return vector embeddings for a list of texts.
    - openai: uses text-embedding-3-* models
    - local: fallback deterministic vectors (dev only)
    """
    if provider == "openai":
        try:
            from openai import OpenAI

            client = OpenAI(api_key=openai_api_key) if openai_api_key else OpenAI()
            resp = client.embeddings.create(model=model, input=texts)
            return [d.embedding for d in resp.data]
        except Exception:
            # Fall through to local if OpenAI not available in env/runtime
            pass

    # local/dev fallback: deterministic hash-based embeddings
    dim = 384
    vecs: List[List[float]] = []
    for t in texts:
        h = hashlib.sha256(t.encode('utf-8')).digest()
        # Expand hash to dim floats in [0,1)
        vals: List[float] = []
        pool = h
        while len(vals) < dim:
            pool = hashlib.sha256(pool).digest()
            for i in range(0, len(pool), 4):
                chunk = int.from_bytes(pool[i:i+4], 'big', signed=False)
                vals.append((chunk % 100000) / 100000.0)
                if len(vals) == dim:
                    break
        vecs.append(vals)
    return vecs
