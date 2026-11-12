"""End-to-end grounding quality tests against the real knowledge base (all 16
Aisling Bank docs). Tests 1, 3, 4, 5, 6 work with the stub fallback (no API
key needed). Tests 2 and 7 need the real LLM's judgement (SEPA Instant vs
standard disambiguation, and recognizing a personal loan isn't a mortgage)
and are skipped unless get_llm_client() would actually return a non-stub
provider — i.e. whichever of LLM_PROVIDER's key (Groq/Gemini/Anthropic) is
set, per the task's "skip when no API key" instruction."""
import re

import pytest

from app.rag.chunking import chunk_documents
from app.rag.embeddings import EmbeddingModel
from app.rag.llm_client import StubLLMClient, get_llm_client
from app.rag.pipeline import NOT_FOUND_MESSAGE, answer_query
from app.rag.vector_store import InMemoryVectorStore
from scripts.index_kb import load_kb_docs

REASON_CODES = [
    "high_transfer_frequency",
    "stable_balance_growth",
    "frequent_spender_eligible_income",
    "established_tenure_eligible_income",
    "digital_channel_preference",
    "high_txn_volume_overdraft_risk",
    "young_customer_debt_optimization",
    "long_term_wealth_building_age_appropriate",
]

requires_real_llm = pytest.mark.skipif(
    isinstance(get_llm_client(), StubLLMClient), reason="needs a configured LLM provider API key"
)


@pytest.fixture(scope="module")
def real_kb_store():
    embedding_model = EmbeddingModel(model_name="__force_hash_fallback__/does-not-exist")
    assert embedding_model.backend == "hash-fallback"

    docs = load_kb_docs()
    chunks = chunk_documents(docs)
    embeddings = embedding_model.encode([c.text for c in chunks])
    store = InMemoryVectorStore()
    store.upsert(chunks, embeddings)
    return embedding_model, store


def _ask_stub(real_kb_store, query: str):
    embedding_model, store = real_kb_store
    return answer_query(
        query, embedding_model=embedding_model, llm_client=StubLLMClient(), vector_store=store
    )


def _ask_real(real_kb_store, query: str):
    embedding_model, store = real_kb_store
    return answer_query(
        query, embedding_model=embedding_model, llm_client=get_llm_client(), vector_store=store
    )


# 1. "how can I send 1000 to my dad" -> one clear method, fee and speed, <60 words, no "#"
def test_send_1000_names_method_with_fee_and_speed(real_kb_store):
    result = _ask_stub(real_kb_store, "how can I send 1000 to my dad")
    assert "#" not in result.answer
    assert len(result.answer.split()) < 60
    assert "€" in result.answer
    assert result.answer != NOT_FOUND_MESSAGE


# 2. "what's the fastest way to send money to my sister" -> SEPA Instant, not standard SEPA
@requires_real_llm
def test_fastest_transfer_picks_sepa_instant_not_standard(real_kb_store):
    result = _ask_real(real_kb_store, "what's the fastest way to send money to my sister")
    assert "seconds" in result.answer.lower() or "instant" in result.answer.lower()
    assert "next business day" not in result.answer.lower()


# 3. "how much loan can I get" -> amount range; sources only the personal loan doc
def test_loan_amount_range_sources_only_personal_loan(real_kb_store):
    result = _ask_stub(real_kb_store, "how much loan can I get")
    assert "1,000" in result.answer or "1000" in result.answer
    assert "50,000" in result.answer
    assert result.sources == ["Personal Loan"]


# 4. "how much does a wire transfer cost" -> €25 domestic and €45 international
def test_wire_transfer_cost_mentions_domestic_and_international_fees(real_kb_store):
    result = _ask_stub(real_kb_store, "how much does a wire transfer cost")
    assert "25" in result.answer
    assert "45" in result.answer


# 5. "can I deposit a cheque from my phone" -> mobile cheque deposit limits and availability
def test_mobile_cheque_deposit_question_is_grounded(real_kb_store):
    result = _ask_stub(real_kb_store, "can I deposit a cheque from my phone")
    assert result.sources == ["Mobile Cheque Deposit"]
    assert "5,000" in result.answer or "10,000" in result.answer


# 6. "do you offer travel insurance" -> exact fallback, no sources
def test_travel_insurance_returns_exact_fallback_no_sources(real_kb_store):
    result = _ask_stub(real_kb_store, "do you offer travel insurance")
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []


# 7. a question about a product that doesn't exist -> exact fallback, no sources.
# ("mortgage rate" used to be this case before a Home Loan product existed —
# now it has a real answer; see tests/test_product_catalog.py for that and
# the other not-covered categories, e.g. travel/car/pet insurance, crypto.)
@requires_real_llm
def test_car_insurance_question_returns_exact_fallback_no_sources(real_kb_store):
    result = _ask_real(real_kb_store, "do you offer car insurance")
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []


# "how do I send money fast" -> numbered steps, max 4
def test_how_do_i_send_money_returns_numbered_steps_max_4(real_kb_store):
    result = _ask_stub(real_kb_store, "how do I send money fast")
    lines = [line for line in result.answer.split("\n") if line.strip()]
    assert 1 <= len(lines) <= 4
    assert all(re.match(r"^\d+\.\s", line) for line in lines)
    assert "#" not in result.answer
    assert "**" not in result.answer


# 8. no answer ever contains "#", "**", "check" (US spelling), "ACH", "$", or a raw reason code
@pytest.mark.parametrize(
    "query",
    [
        "how can I send 1000 to my dad",
        "how much loan can I get",
        "how much does a wire transfer cost",
        "can I deposit a cheque from my phone",
        "do you offer travel insurance",
        "what is your mortgage rate for a second home",
    ],
)
def test_no_answer_contains_markdown_us_terms_or_reason_codes(real_kb_store, query):
    result = _ask_stub(real_kb_store, query)
    assert "#" not in result.answer
    assert "**" not in result.answer
    assert "$" not in result.answer
    assert not re.search(r"\bACH\b", result.answer)
    assert "check" not in result.answer.lower()
    for code in REASON_CODES:
        assert code not in result.answer


def test_no_heading_text_glued_to_a_sentence(real_kb_store):
    """TASK 2 regression: a document's own title/heading must never appear
    fused onto the start of a sentence (the original "Wire Transfer Fees A
    wire transfer is..." bug)."""
    for query in ["how much does a wire transfer cost", "how much loan can I get"]:
        result = _ask_stub(real_kb_store, query)
        for title in ("Wire Transfer Fees", "Personal Loan", "SEPA Credit Transfer"):
            assert f"{title} A" not in result.answer
            assert f"{title} The" not in result.answer


def test_ach_ira_and_dollar_sign_removed_from_kb():
    for doc in load_kb_docs():
        assert "$" not in doc["text"]
        assert not re.search(r"\bACH\b", doc["text"])
        assert not re.search(r"\bIRA\b", doc["text"])
        assert "401k" not in doc["text"].lower()
        assert "United States" not in doc["text"]
        assert "check" not in doc["text"].lower()
