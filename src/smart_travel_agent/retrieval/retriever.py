from pathlib import Path

import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_core.tools import tool

from smart_travel_agent.config import get_embeddings
from smart_travel_agent.retrieval.build_index import INDEX_DIR


def _load_index() -> FAISS:
    if not Path(INDEX_DIR).exists():
        raise FileNotFoundError(f"No FAISS index found at {INDEX_DIR}. Run build_index.py first.")

    return FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def search(query: str, k: int = 2) -> list[tuple[str, float, dict]]:
    """Embeds the query, then ranks the index's already-embedded chunks by cosine similarity.

    Every chunk's embedding was generated once, at build time, by OpenAIEmbeddings (inside
    FAISS.from_documents) and is stored as a dense vector in the FAISS index — this is
    semantic vector search, not a plain-text/keyword match. Only the query gets embedded here;
    the document vectors are read back out of the index and ranked by cosine similarity.
    """
    faiss_index = _load_index()

    query_vector = np.array(faiss_index.embedding_function.embed_query(query), dtype="float32")

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


@tool
def search_school_calendar(query: str) -> str:
    """Search the indexed school calendar documents for dates matching the query
    (e.g. "spring break", "winter break", "holidays in March") and return the top
    matching excerpts, each tagged with its source school and similarity score."""
    results = search(query, k=5)
    return "\n\n---\n\n".join(
        f"[{metadata.get('school', 'unknown school')}] (score={score:.3f}) {text}"
        for text, score, metadata in results
    )
