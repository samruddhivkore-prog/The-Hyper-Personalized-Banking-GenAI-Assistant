from app.rag.chunking import chunk_documents
from app.rag.embeddings import EmbeddingModel
from app.rag.llm_client import StubLLMClient
from app.rag.pipeline import NOT_FOUND_MESSAGE, answer_query
from app.rag.vector_store import InMemoryVectorStore

RTP_DOC = {
    "doc_id": "real_time_payments_faq",
    "source_title": "Real-Time Payments FAQ",
    "text": (
        "# Real-Time Payments FAQ\n\n"
        "Real-Time Payments let you send money in seconds, any day of the week.\n\n"
        "## How much does it cost?\n"
        "Each Real-Time Payment costs a flat fee of €1.00 per transfer.\n\n"
        "## What is the transfer limit?\n"
        "You can send up to €2,500 per transaction."
    ),
}
WIRE_DOC = {
    "doc_id": "wire_transfer_fees",
    "source_title": "Wire Transfer Fees",
    "text": (
        "# Wire Transfer Fees\n\n"
        "An outgoing domestic wire transfer costs €25.00 per transfer.\n\n"
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


def test_answer_query_grounds_answer_in_retrieved_chunk_and_strips_markdown():
    embedding_model = _hash_only_embedding_model()
    store = _build_store([RTP_DOC, WIRE_DOC], embedding_model)

    result = answer_query(
        query="What's the fastest way to send €2,000 to my sister's account?",
        embedding_model=embedding_model,
        llm_client=StubLLMClient(),
        vector_store=store,
    )

    assert result.confidence in {"high", "medium"}
    assert "Real-Time Payments FAQ" in result.sources
    assert "€1.00" in result.answer or "seconds" in result.answer
    assert "#" not in result.answer
    assert result.retrieved_chunks


def test_answer_query_degrades_gracefully_when_doc_deleted():
    """The core grounding test from the spec: delete the one relevant source
    document and confirm the system says it doesn't know instead of guessing."""
    embedding_model = _hash_only_embedding_model()
    store = _build_store([RTP_DOC], embedding_model)

    store.delete_by_doc_id("real_time_payments_faq")

    result = answer_query(
        query="What's the fastest way to send €2,000 to my sister's account?",
        embedding_model=embedding_model,
        llm_client=StubLLMClient(),
        vector_store=store,
    )

    assert result.confidence == "none"
    assert result.sources == []
    assert result.retrieved_chunks == []
    assert result.answer == NOT_FOUND_MESSAGE


def test_answer_query_empty_store_returns_no_info():
    embedding_model = _hash_only_embedding_model()
    store = InMemoryVectorStore()

    result = answer_query(
        query="anything",
        embedding_model=embedding_model,
        llm_client=StubLLMClient(),
        vector_store=store,
    )
    assert result.confidence == "none"
    assert result.answer == NOT_FOUND_MESSAGE


def test_answer_query_rejects_irrelevant_question_even_with_nonempty_store():
    """A question with zero meaningful word overlap with any retrieved chunk
    must fall back, even though the store isn't empty — the relevance
    threshold from Problem 1."""
    embedding_model = _hash_only_embedding_model()
    store = _build_store([RTP_DOC, WIRE_DOC], embedding_model)

    result = answer_query(
        query="Do you offer pet insurance for my goldfish?",
        embedding_model=embedding_model,
        llm_client=StubLLMClient(),
        vector_store=store,
    )

    assert result.confidence == "none"
    assert result.sources == []
    assert result.answer == NOT_FOUND_MESSAGE
