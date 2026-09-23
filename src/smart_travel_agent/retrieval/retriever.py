import functools
from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

from smart_travel_agent.config import get_embeddings
from smart_travel_agent.retrieval.build_index import INDEX_DIR


@functools.lru_cache(maxsize=1)
def _load_index() -> FAISS:
    """Cached after the first call — the index only changes via an explicit, offline
    `build_index.py` rerun, never while the app is running, so reloading it from disk and
    spinning up a fresh embeddings client on every search_school_calendar call (calendar_keeper
    alone makes at least 2 such calls per run, one per school) is pure repeated cost."""
    if not Path(INDEX_DIR).exists():
        raise FileNotFoundError(f"No FAISS index found at {INDEX_DIR}. Run build_index.py first.")

    return FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


@functools.lru_cache(maxsize=256)
def _embed_query(query: str) -> tuple[float, ...]:
    """Cached per distinct query string — real, billed OpenAI embedding calls, and
    calendar_keeper's own queries repeat heavily (the same break names recur across trips
    and users). Returns a tuple rather than a list/ndarray so the cached value is immutable
    and a caller mutating its copy can't corrupt what's shared across cache hits."""
    return tuple(_load_index().embedding_function.embed_query(query))


def search(query: str, k: int = 2) -> list[tuple[str, float, dict]]:
    """Embeds the query, then ranks the index's already-embedded chunks by cosine similarity.

    Every chunk's embedding was generated once, at build time, by OpenAIEmbeddings (inside
    FAISS.from_documents) and is stored as a dense vector in the FAISS index — this is
    semantic vector search, not a plain-text/keyword match. Only the query gets embedded here;
    the document vectors are read back out of the index and ranked by cosine similarity.
    """
    faiss_index = _load_index()

    query_vector = np.array(_embed_query(query), dtype="float32")

    ntotal = faiss_index.index.ntotal
    doc_vectors = faiss_index.index.reconstruct_n(0, ntotal)

    query_unit = query_vector / np.linalg.norm(query_vector)
    doc_units = doc_vectors / np.linalg.norm(doc_vectors, axis=1, keepdims=True)
    scores = doc_units @ query_unit

    top_k_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_k_indices:
        doc_id = faiss_index.index_to_docstore_id[int(idx)]
        doc = faiss_index.docstore._dict[doc_id]
        results.append((doc.page_content, float(scores[idx]), doc.metadata))
    return results


@functools.lru_cache(maxsize=256)
def _search_school_calendar_cached(query: str) -> str:
    """Cached per exact query string — calendar_keeper issues the same handful of queries
    (e.g. "Stratford Thanksgiving break") over and over across different trips/users, and
    besides the embedding call, re-running the FAISS similarity math and reformatting the
    result is redundant work for an identical query against an index that doesn't change."""
    results = search(query, k=5)
    print(f"Sarita - Found results for query: {query}")
    return "\n\n---\n\n".join(
        f"[{metadata.get('school', 'unknown school')}] (score={score:.3f}) {text}"
        for text, score, metadata in results
    )


@tool
def search_school_calendar(query: str) -> str:
    """Search the indexed school calendar documents for dates matching the query
    (e.g. "spring break", "winter break", "holidays in March") and return the top
    matching excerpts, each tagged with its source school and similarity score."""
    print(f"Sarita - Searching school calendar for query: {query}")
    return _search_school_calendar_cached(query)
