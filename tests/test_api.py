from app.rag.chunking import chunk_documents
from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import InMemoryVectorStore

RTP_DOC = {
    "doc_id": "real_time_payments_faq",
    "source_title": "Real-Time Payments FAQ",
    "text": (
        "Real-Time Payments let you send money in seconds, any day of the week. "
        "Each Real-Time Payment costs a flat fee of $1.00 per transfer."
    ),
}


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["database"] is True


def test_recommendations_404_for_unknown_customer(client):
    resp = client.post("/recommendations", json={"customer_id": "DOES_NOT_EXIST"})
    assert resp.status_code == 404


def test_recommendations_without_query_returns_template_summary(client, seeded_customer):
    resp = client.post("/recommendations", json={"customer_id": seeded_customer.customer_id})
    assert resp.status_code == 200
    body = resp.json()
    assert body["customer_id"] == seeded_customer.customer_id
    assert body["segment"] == seeded_customer.segment_label
    assert isinstance(body["recommendations"], list)
    assert body["assistant_response"]["sources"] == []
    # fairness_flags reflects whatever the committed reports/fairness_sample_*.json
    # artifacts say — see app/fairness.py; just check the shape, not emptiness.
    assert isinstance(body["fairness_flags"], list)
    assert "timestamp" in body


def test_recommendations_with_query_is_grounded(client, seeded_customer, monkeypatch):
    embedding_model = EmbeddingModel(model_name="__force_hash_fallback__/does-not-exist")
    assert embedding_model.backend == "hash-fallback"

    store = InMemoryVectorStore()
    chunks = chunk_documents([RTP_DOC])
    embeddings = embedding_model.encode([c.text for c in chunks])
    store.upsert(chunks, embeddings)

    monkeypatch.setattr("app.rag.pipeline.get_embedding_model", lambda: embedding_model)
    monkeypatch.setattr("app.rag.pipeline.get_vector_store", lambda: store)

    resp = client.post(
        "/recommendations",
        json={
            "customer_id": seeded_customer.customer_id,
            "query": "What's the fastest way to send money?",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["assistant_response"]["confidence"] in {"high", "medium", "low"}
    assert "Real-Time Payments FAQ" in body["assistant_response"]["sources"]


def test_recommendations_with_query_degrades_when_no_context(client, seeded_customer, monkeypatch):
    embedding_model = EmbeddingModel(model_name="__force_hash_fallback__/does-not-exist")
    empty_store = InMemoryVectorStore()

    monkeypatch.setattr("app.rag.pipeline.get_embedding_model", lambda: embedding_model)
    monkeypatch.setattr("app.rag.pipeline.get_vector_store", lambda: empty_store)

    resp = client.post(
        "/recommendations",
        json={"customer_id": seeded_customer.customer_id, "query": "anything at all"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["assistant_response"]["confidence"] == "none"
    assert "don't have enough information" in body["assistant_response"]["answer"].lower()


def test_segments_endpoint_lists_seeded_segment(client, seeded_customer):
    resp = client.get("/segments")
    assert resp.status_code == 200
    body = resp.json()
    assert any(s["segment_id"] == seeded_customer.segment_id for s in body)
