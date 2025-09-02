from typing import List, Tuple
import numpy as np
from ..adapters.embeddings import embed_texts
from ..models.chunk import Chunk


class SimpleRetriever:
    """Retriever scaffold.
    Uses embeddings if available; otherwise a deterministic fallback.
    """

    def __init__(self, chunks: List[Chunk], emb_provider: str = "openai", emb_model: str = "text-embedding-3-small", openai_api_key: str | None = None):
        self.chunks = chunks
        self.emb_provider = emb_provider
        self.emb_model = emb_model
        self.openai_api_key = openai_api_key
        self.chunk_texts = [c.text for c in chunks]
        self.embeddings = embed_texts(self.chunk_texts, provider=self.emb_provider, model=self.emb_model, openai_api_key=self.openai_api_key)

    def query(self, q: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        try:
            import faiss  # type: ignore

            q_vec = embed_texts([q], provider=self.emb_provider, model=self.emb_model, openai_api_key=self.openai_api_key)[0]
            q_vec = np.array([q_vec], dtype="float32")
            faiss.normalize_L2(q_vec)
            vecs = np.array(self.embeddings, dtype="float32")
            faiss.normalize_L2(vecs)
            index = faiss.IndexFlatIP(vecs.shape[1])
            index.add(vecs)
            scores, idxs = index.search(q_vec, top_k)
            out: List[Tuple[Chunk, float]] = []
            for i, score in zip(idxs[0], scores[0]):
                if i == -1:
                    continue
                out.append((self.chunks[int(i)], float(score)))
            return out
        except Exception:
            ranked = sorted(self.chunks, key=lambda c: len(c.text), reverse=True)
            return [(c, 1.0) for c in ranked[:top_k]]


class FaissRetriever:
    def __init__(self, chunks: List[Chunk], index: "faiss.Index", emb_provider: str, emb_model: str, openai_api_key: str | None):
        self.chunks = chunks
        self.index = index
        self.emb_provider = emb_provider
        self.emb_model = emb_model
        self.openai_api_key = openai_api_key

    def query(self, q: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        import faiss  # type: ignore

        q_vec = embed_texts([q], provider=self.emb_provider, model=self.emb_model, openai_api_key=self.openai_api_key)[0]
        import numpy as np

        q_vec = np.array([q_vec], dtype="float32")
        faiss.normalize_L2(q_vec)
        scores, idxs = self.index.search(q_vec, top_k)
        out: List[Tuple[Chunk, float]] = []
        for i, score in zip(idxs[0], scores[0]):
            if i == -1:
                continue
            out.append((self.chunks[int(i)], float(score)))
        return out
