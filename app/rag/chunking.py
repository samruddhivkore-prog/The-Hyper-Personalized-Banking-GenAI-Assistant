"""Document chunking for the RAG knowledge base.

Chunks are sized in words (a rough proxy for the ~300-500 token target in the
spec — good enough for short synthetic policy docs) with overlap so that
facts spanning a chunk boundary aren't lost.
"""
from dataclasses import dataclass


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    source_title: str
    text: str


def chunk_document(
    doc_id: str,
    source_title: str,
    text: str,
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> list[Chunk]:
    words = text.split()
    if not words:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    step = max(1, chunk_size_words - overlap_words)
    while start < len(words):
        end = min(start + chunk_size_words, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}::chunk{idx}",
                source_title=source_title,
                text=chunk_text,
            )
        )
        idx += 1
        if end == len(words):
            break
        start += step
    return chunks


def chunk_documents(docs: list[dict]) -> list[Chunk]:
    """docs: [{"doc_id": ..., "source_title": ..., "text": ...}, ...]"""
    all_chunks: list[Chunk] = []
    for doc in docs:
        all_chunks.extend(
            chunk_document(doc["doc_id"], doc["source_title"], doc["text"])
        )
    return all_chunks
