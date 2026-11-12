from app.rag.chunking import chunk_documents
from app.rag.embeddings import EmbeddingModel
from app.rag.pipeline import NOT_FOUND_MESSAGE
from app.rag.vector_store import InMemoryVectorStore

TRANSFER_DOC = {
    "doc_id": "real_time_payments_faq",
    "source_title": "Real-Time Payments FAQ",
    "text": (
        "Real-Time Payments let you send money in seconds, any day of the week. "
        "Each Real-Time Payment costs a flat fee of €1.00 per transfer."
    ),
}


def _patch_rag(monkeypatch, docs):
    embedding_model = EmbeddingModel(model_name="__force_hash_fallback__/does-not-exist")
    assert embedding_model.backend == "hash-fallback"

    store = InMemoryVectorStore()
    if docs:
        chunks = chunk_documents(docs)
        embeddings = embedding_model.encode([c.text for c in chunks])
        store.upsert(chunks, embeddings)

    monkeypatch.setattr("app.rag.pipeline.get_embedding_model", lambda: embedding_model)
    monkeypatch.setattr("app.rag.pipeline.get_vector_store", lambda: store)
    return store


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
    for rec in body["recommendations"]:
        assert rec["plain_text"]
        assert rec["learn_more"]
    assert body["assistant_response"]["sources"] == []
    # fairness_flags reflects whatever the committed reports/fairness_sample_*.json
    # artifacts say — see app/fairness.py; just check the shape, not emptiness.
    assert isinstance(body["fairness_flags"], list)
    assert "timestamp" in body


def test_recommendations_with_query_is_grounded(client, seeded_customer, monkeypatch):
    _patch_rag(monkeypatch, [TRANSFER_DOC])

    resp = client.post(
        "/recommendations",
        json={
            "customer_id": seeded_customer.customer_id,
            "query": "What's the fastest way to send money?",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["assistant_response"]["confidence"] in {"high", "medium"}
    assert "Real-Time Payments FAQ" in body["assistant_response"]["sources"]


def test_recommendations_with_query_degrades_when_no_context(client, seeded_customer, monkeypatch):
    _patch_rag(monkeypatch, [])

    resp = client.post(
        "/recommendations",
        json={"customer_id": seeded_customer.customer_id, "query": "anything at all"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["assistant_response"]["confidence"] == "none"
    assert body["assistant_response"]["answer"] == NOT_FOUND_MESSAGE


def test_segments_endpoint_lists_seeded_segment(client, seeded_customer):
    resp = client.get("/segments")
    assert resp.status_code == 200
    body = resp.json()
    assert any(s["segment_id"] == seeded_customer.segment_id for s in body)


# ---- Problem-specific acceptance tests ----


def test_chat_transfer_question_uses_transfer_policy_chunk(client, seeded_customer, monkeypatch):
    """Acceptance test 1: a transfer question returns an answer grounded in the
    transfer policy chunk, with a readable source name."""
    _patch_rag(monkeypatch, [TRANSFER_DOC])

    resp = client.post(
        "/chat",
        json={
            "customer_id": seeded_customer.customer_id,
            "query": "What's the fastest way to send money to my sister?",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    answer = body["assistant_response"]["answer"]
    assert body["assistant_response"]["sources"] == ["Real-Time Payments FAQ"]
    assert "seconds" in answer or "€1.00" in answer


def test_chat_question_with_no_matching_policy_returns_fallback(client, seeded_customer, monkeypatch):
    """Acceptance test 2: a question with no matching policy returns the
    fallback message, never an invented answer."""
    _patch_rag(monkeypatch, [TRANSFER_DOC])

    resp = client.post(
        "/chat",
        json={"customer_id": seeded_customer.customer_id, "query": "Do you offer pet insurance?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["assistant_response"]["answer"] == NOT_FOUND_MESSAGE
    assert body["assistant_response"]["sources"] == []


def test_chat_answer_does_not_repeat_the_product_list(client, seeded_customer, monkeypatch):
    """Acceptance test 3: /chat's answer only answers the question — it must
    not repeat the customer's recommendation list (with reason codes/scores)
    or the old "We think X is your best next step" filler inside it. Note:
    this does NOT forbid a product name appearing when it's genuinely part of
    the grounded policy text being quoted (e.g. a policy doc about Real-Time
    Payments will say "Real-Time Payments" — that's a correct answer, not
    list-leakage)."""
    _patch_rag(monkeypatch, [TRANSFER_DOC])

    resp = client.post(
        "/chat",
        json={
            "customer_id": seeded_customer.customer_id,
            "query": "What's the fastest way to send money?",
        },
    )
    body = resp.json()
    answer = body["assistant_response"]["answer"]

    for rec in body["recommendations"]:
        assert rec["reason_code"] not in answer
    assert "score=" not in answer
    assert "best next step" not in answer.lower()


def test_customer_facing_text_has_no_internal_jargon(client, seeded_customer, monkeypatch):
    """Acceptance test 4: the customer-facing text (assistant answer + card
    copy) never contains a raw reason_code, a score, or the segment label —
    those stay in the raw API fields for the Admin view only."""
    _patch_rag(monkeypatch, [TRANSFER_DOC])

    resp = client.post(
        "/recommendations",
        json={
            "customer_id": seeded_customer.customer_id,
            "query": "What's the fastest way to send money?",
        },
    )
    body = resp.json()
    customer_facing_text = body["assistant_response"]["answer"] + " " + " ".join(
        rec["plain_text"] + " " + rec["learn_more"] for rec in body["recommendations"]
    )

    for rec in body["recommendations"]:
        assert rec["reason_code"] not in customer_facing_text
    if body["segment"]:
        assert body["segment"] not in customer_facing_text
    assert "score=" not in customer_facing_text
    assert "demographic parity" not in customer_facing_text


def test_admin_fields_present_in_raw_response_for_frontend_admin_tab(client, seeded_customer, monkeypatch):
    """Acceptance test 5: everything the Admin tab needs (segment, raw scores,
    reason codes, retrieval chunks, confidence, fairness flags) is present in
    the raw API response — the Admin tab just renders fields already there,
    gated by SHOW_ADMIN in the frontend, not by the backend."""
    _patch_rag(monkeypatch, [TRANSFER_DOC])

    resp = client.post(
        "/chat",
        json={
            "customer_id": seeded_customer.customer_id,
            "query": "What's the fastest way to send money?",
        },
    )
    body = resp.json()
    assert "segment" in body
    assert all({"score", "reason_code"} <= rec.keys() for rec in body["recommendations"])
    assert "retrieved_chunks" in body["assistant_response"]
    assert body["assistant_response"]["retrieved_chunks"]
    assert "confidence" in body["assistant_response"]
    assert "fairness_flags" in body
