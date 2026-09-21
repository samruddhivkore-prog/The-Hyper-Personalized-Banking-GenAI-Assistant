"""Chunk + embed + upsert the knowledge base documents into the vector store
(spec section 8.1, step 9). Re-run any time data/kb_docs/ changes.

Usage: python -m scripts.index_kb
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.logging_config import configure_logging, get_logger
from app.rag.chunking import chunk_documents
from app.rag.embeddings import get_embedding_model
from app.rag.vector_store import get_vector_store

logger = get_logger(__name__)

KB_DIR = Path(__file__).resolve().parent.parent / "data" / "kb_docs"


def _title_from_filename(path: Path) -> str:
    return path.stem.replace("_", " ").title()


def load_kb_docs() -> list[dict]:
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        # first markdown heading, if present, makes a nicer source title
        first_line = text.strip().splitlines()[0] if text.strip() else ""
        title = first_line.lstrip("# ").strip() if first_line.startswith("#") else _title_from_filename(path)
        docs.append({"doc_id": path.stem, "source_title": title, "text": text})
    return docs


def index_kb() -> int:
    configure_logging()
    docs = load_kb_docs()
    chunks = chunk_documents(docs)

    embedding_model = get_embedding_model()
    vector_store = get_vector_store()

    texts = [c.text for c in chunks]
    embeddings = embedding_model.encode(texts)
    vector_store.upsert(chunks, embeddings)

    logger.info(
        "kb_indexed",
        documents=len(docs),
        chunks=len(chunks),
        embedding_backend=embedding_model.backend,
    )
    return len(chunks)


def remove_kb_doc(doc_id: str) -> None:
    """Removes a single document's chunks from the vector store — used to
    demonstrate graceful degradation (README's 'delete a doc' test)."""
    configure_logging()
    vector_store = get_vector_store()
    vector_store.delete_by_doc_id(doc_id)
    logger.info("kb_doc_removed", doc_id=doc_id)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--remove", help="doc_id to remove instead of (re)indexing everything")
    args = parser.parse_args()

    if args.remove:
        remove_kb_doc(args.remove)
    else:
        index_kb()
