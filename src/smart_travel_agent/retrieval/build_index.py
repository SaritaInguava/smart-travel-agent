from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from smart_travel_agent.config import get_embeddings
from smart_travel_agent.retrieval.chunking import chunk_calendar_by_week, merge_consecutive_events
from smart_travel_agent.retrieval.harker_scraper import fetch_school_year_events

DATA_DIR = Path(__file__).parent.parent / "data"
INDEX_DIR = Path(__file__).parent / "faiss_index"


def _load_stratford_docs(source_dir: Path = DATA_DIR, glob: str = "**/*.pdf") -> list[Document]:
    loader = DirectoryLoader(str(source_dir), glob=glob)
    documents = loader.load()
    for doc in documents:
        doc.metadata["school"] = "Stratford"
    return documents


def _load_harker_docs() -> list[Document]:
    events = merge_consecutive_events(fetch_school_year_events())
    chunks = chunk_calendar_by_week(events)
    return [
        Document(
            page_content=chunk["text"],
            metadata={
                "school": "Harker Upper School",
                "time_window": chunk["time_window"],
                "start_date": chunk["start_date"],
                "end_date": chunk["end_date"],
                "event_count": chunk["event_count"],
            },
        )
        for chunk in chunks
    ]


def build_index() -> None:
    """Rebuilds the FAISS index from scratch across all school calendar sources.

    There's no in-place metadata update for FAISS-backed documents, so adding a new
    source (or new metadata to an existing one) means rebuilding the whole index rather
    than patching individual entries.
    """
    embeddings = get_embeddings()

    documents = _load_stratford_docs() + _load_harker_docs()
    if not documents:
        raise ValueError("No documents found to index.")

    faiss_index = FAISS.from_documents(documents, embeddings)
    vector_dim = faiss_index.index.d
    print(f"Embedded {len(documents)} chunk(s) into {vector_dim}-dimensional vectors.")
    for doc in documents:
        print(f"  [{doc.metadata.get('school')}] {len(doc.page_content)} chars")

    faiss_index.save_local(str(INDEX_DIR))
    print(f"Saved FAISS index to {INDEX_DIR}")


if __name__ == "__main__":
    build_index()
