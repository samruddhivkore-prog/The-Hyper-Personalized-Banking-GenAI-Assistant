"""PART 5 tests: which_product routing against the real product catalog.
These need genuine semantic matching (the stub's keyword fallback can't
reliably distinguish "loan for a car" from "loan for a house"), so the
"should match" cases are real-LLM tests, skipped when no provider key is
set. The "not covered" cases are tested with the stub too, since the
relevance gate alone is enough to reject them without any LLM judgement."""
import pytest

from app.rag.llm_client import StubLLMClient, get_llm_client
from app.rag.pipeline import NOT_FOUND_MESSAGE, answer_query
from app.services.product_match import match_product

requires_real_llm = pytest.mark.skipif(
    isinstance(get_llm_client(), StubLLMClient), reason="needs a configured LLM provider API key"
)


@requires_real_llm
@pytest.mark.parametrize(
    "query,expected_source",
    [
        ("which loan can i take to buy a car", "Personal Loan"),
        ("loan type for house", "Home Loan"),
        ("what is your mortgage rate for a second home", "Home Loan"),
        ("loan to pay my university fees", "Student Loan"),
        ("which account is best for saving", "High-Yield Savings"),
        ("which card should i get", "Rewards Credit Card"),
    ],
)
def test_which_product_matches_only_the_right_source(query, expected_source):
    llm = get_llm_client()
    result = match_product(query, llm)
    assert result["sources"] == [expected_source]


@requires_real_llm
def test_how_do_i_open_an_account_returns_steps_from_opening_an_account():
    result = answer_query("how do i open a new account")
    assert result.sources == ["Opening an Account"]
    lines = [line for line in result.answer.split("\n") if line.strip()]
    assert 1 <= len(lines) <= 4


@pytest.mark.parametrize(
    "query",
    [
        "do you offer travel insurance",
        "do you offer car insurance",
        "do you offer pet insurance",
        "can i trade crypto with you",
    ],
)
def test_not_covered_categories_return_exact_fallback_no_sources(query):
    result = answer_query(query)
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []


def test_student_loan_and_refinance_are_kept_distinct_in_catalog():
    """PART 1: the new Student Loan (new borrowing) and existing Student Loan
    Refinance must stay separate catalog entries with different purposes."""
    from app.services.product_match import load_catalog

    catalog = {p["name"]: p for p in load_catalog()}
    assert "Student Loan" in catalog
    assert "Student Loan Refinance" in catalog
    assert catalog["Student Loan"]["policy_doc"] != catalog["Student Loan Refinance"]["policy_doc"]
    assert "tuition" in " ".join(catalog["Student Loan"]["purposes"]).lower()
    assert "refinanc" in " ".join(catalog["Student Loan Refinance"]["purposes"]).lower()


def test_personal_loan_cannot_be_used_for_property():
    from scripts.index_kb import load_kb_docs

    doc = next(d for d in load_kb_docs() if d["doc_id"] == "personal_loan_eligibility")
    assert "cannot be used to buy or build a property" in doc["text"].lower()
