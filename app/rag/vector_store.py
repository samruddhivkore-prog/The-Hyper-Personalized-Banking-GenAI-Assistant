"""Vector store for the RAG knowledge base.

Backed by Chroma (persistent, local, free) when available. Falls back to a
tiny in-memory cosine-similarity store when chromadb can't be imported, so
unit tests and constrained environments don't need it installed.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from app.config import get_settings
from app.rag.chunking import Chunk


@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: str
    source_title: str
    text: str
    score: float


class InMemoryVectorStore:
    def __init__(self):
        self._ids: list[str] = []
        self._embeddings: list[np.ndarray] = []
        self._metadatas: list[dict] = []
        self._documents: list[str] = []

    def upsert(self, chunks: list[Chunk], embeddings: list[np.ndarray]) -> None:
        for chunk, emb in zip(chunks, embeddings):
            if chunk.chunk_id in self._ids:
                idx = self._ids.index(chunk.chunk_id)
                self._embeddings[idx] = emb
                self._documents[idx] = chunk.text
                self._metadatas[idx] = {"doc_id": chunk.doc_id, "source_title": chunk.source_title}
            else:
                self._ids.append(chunk.chunk_id)
                self._embeddings.append(emb)
                self._documents.append(chunk.text)
                self._metadatas.append({"doc_id": chunk.doc_id, "source_title": chunk.source_title})

    def delete_by_doc_id(self, doc_id: str) -> None:
        keep = [i for i, m in enumerate(self._metadatas) if m["doc_id"] != doc_id]
        self._ids = [self._ids[i] for i in keep]
        self._embeddings = [self._embeddings[i] for i in keep]
        self._documents = [self._documents[i] for i in keep]
        self._metadatas = [self._metadatas[i] for i in keep]

    def count(self) -> int:
        return len(self._ids)

    def similarity_search(self, query_embedding: np.ndarray, k: int = 4) -> list[RetrievedChunk]:
        if not self._embeddings:
            return []
        sims = []
        for emb in self._embeddings:
            denom = (np.linalg.norm(query_embedding) * np.linalg.norm(emb)) or 1e-9
            sims.append(float(np.dot(query_embedding, emb) / denom))
        order = np.argsort(sims)[::-1][:k]
        results = []
        for i in order:
            results.append(
                RetrievedChunk(
                    doc_id=self._metadatas[i]["doc_id"],
                    chunk_id=self._ids[i],
                    source_title=self._metadatas[i]["source_title"],
                    text=self._documents[i],
                    score=sims[i],
                )
            )
        return results


class ChromaVectorStore:
    def __init__(self, persist_dir: str, collection_name: str = "kb_docs"):
        import chromadb

        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(collection_name)

    def upsert(self, chunks: list[Chunk], embeddings: list[np.ndarray]) -> None:
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=[e.tolist() for e in embeddings],
            documents=[c.text for c in chunks],
            metadatas=[{"doc_id": c.doc_id, "source_title": c.source_title} for c in chunks],
        )

    def delete_by_doc_id(self, doc_id: str) -> None:
        self._collection.delete(where={"doc_id": doc_id})

    def count(self) -> int:
        return self._collection.count()

    def similarity_search(self, query_embedding: np.ndarray, k: int = 4) -> list[RetrievedChunk]:
        if self.count() == 0:
            return []
        result = self._collection.query(query_embeddings=[query_embedding.tolist()], n_results=k)
        results = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for cid, doc, meta, dist in zip(ids, docs, metas, dists):
            results.append(
                RetrievedChunk(
                    doc_id=meta["doc_id"],
                    chunk_id=cid,
                    source_title=meta["source_title"],
                    text=doc,
                    score=1.0 - float(dist),
                )
            )
        return results


@lru_cache
def get_vector_store():
    settings = get_settings()
    try:
        return ChromaVectorStore(settings.chroma_persist_dir)
    except Exception:
        return InMemoryVectorStore()
