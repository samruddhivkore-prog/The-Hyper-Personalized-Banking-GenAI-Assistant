from app.rag.chunking import chunk_documents
from app.rag.embeddings import EmbeddingModel
from app.rag.llm_client import StubLLMClient
from app.rag.pipeline import answer_query
from app.rag.vector_store import InMemoryVectorStore

RTP_DOC = {
    "doc_id": "real_time_payments_faq",
    "source_title": "Real-Time Payments FAQ",
    "text": (
        "Real-Time Payments let you send money in seconds, any day of the week. "
        "Each Real-Time Payment costs a flat fee of $1.00 per transfer. "
        "You can send up to $2,500 per transaction."
    ),
}
WIRE_DOC = {
    "doc_id": "wire_transfer_fees",
    "source_title": "Wire Transfer Fees",
    "text": (
        "An outgoing domestic wire transfer costs $25.00 per transfer. "
        "Wires submitted before 3pm are typically delivered the same business day."
    ),
}


def _build_store(docs, embedding_model):
    store = InMemoryVectorStore()
    chunks = chunk_documents(docs)
    embeddings = embedding_model.encode([c.text for c in chunks])
    store.upsert(chunks, embeddings)
    return store


def _hash_only_embedding_model() -> EmbeddingModel:
    model = EmbeddingModel(model_name="__force_hash_fallback__/does-not-exist")
    assert model.backend == "hash-fallback"
    return model


def test_answer_query_grounds_answer_in_retrieved_chunk():
    embedding_model = _hash_only_embedding_model()
    store = _build_store([RTP_DOC, WIRE_DOC], embedding_model)

    result = answer_query(
        query="What's the fastest way to send $2,000 to my sister's account?",
        segment="digital_first_young_professional",
        recommendations=[
            {"product": "Real-Time Payments", "score": 0.92, "reason_code": "high_transfer_frequency"}
        ],
        embedding_model=embedding_model,
        llm_client=StubLLMClient(),
        vector_store=store,
    )

    assert result.confidence in {"high", "medium", "low"}
    assert "Real-Time Payments FAQ" in result.sources
    assert "$1.00" in result.answer or "seconds" in result.answer


def test_answer_query_degrades_gracefully_when_doc_deleted():
    """The core grounding test from the spec: delete the one relevant source
    document and confirm the system says it doesn't know instead of guessing."""
    embedding_model = _hash_only_embedding_model()
    store = _build_store([RTP_DOC], embedding_model)

    store.delete_by_doc_id("real_time_payments_faq")

    result = answer_query(
        query="What's the fastest way to send $2,000 to my sister's account?",
        segment="digital_first_young_professional",
        recommendations=[],
        embedding_model=embedding_model,
        llm_client=StubLLMClient(),
        vector_store=store,
    )

    assert result.confidence == "none"
    assert result.sources == []
    assert "don't have enough information" in result.answer.lower()


def test_answer_query_empty_store_returns_no_info():
    embedding_model = _hash_only_embedding_model()
    store = InMemoryVectorStore()

    result = answer_query(
        query="anything",
        segment=None,
        recommendations=[],
        embedding_model=embedding_model,
        llm_client=StubLLMClient(),
        vector_store=store,
    )
    assert result.confidence == "none"
    assert result.answer == "I don't have enough information to answer that."
