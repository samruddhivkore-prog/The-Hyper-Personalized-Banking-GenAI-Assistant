"""Online pipeline: HandleRecommendationRequest (spec section 8.2), extended
with rule-based intent routing for spending questions and card actions —
see app/services/intent.py for why that routing is rule-based, not LLM-based."""
import re
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.fairness import get_fairness_auditor
from app.logging_config import get_logger
from app.models import Customer, InteractionLog, Recommendation
from app.rag.llm_client import get_llm_client
from app.rag.pipeline import NOT_FOUND_MESSAGE, answer_query
from app.recommender import RecommenderModel
from app.services.cards import get_customer_card
from app.services.copy import product_learn_more, reason_code_to_plain_text
from app.services.intent import detect_intent
from app.services.period_parser import default_period, parse_period
from app.services.product_match import match_product
from app.services.question_type import classify_question_type
from app.services.spending import compute_spending_summary, render_spending_sentence
from app.services.templates import render_default_summary

logger = get_logger(__name__)

_recommender = RecommenderModel()

CANT_PARSE_PERIOD = "Sorry, I couldn't understand that time period. Could you rephrase it?"
NO_CARD_ON_FILE = "You don't have a card on file for this account."

_MENTIONS_PRODUCT_CATEGORY = re.compile(
    r"\b(loan|mortgage|account|card|saving|savings|pension|invest)\w*", re.IGNORECASE
)


def _profile_dict(customer: Customer) -> dict:
    return {
        "customer_id": customer.customer_id,
        "age": customer.age,
        "income_bracket": customer.income_bracket,
        "account_tenure_months": customer.account_tenure_months,
        "products_held": customer.products_held or [],
        "channel_preference": customer.channel_preference,
        "avg_monthly_transactions": customer.avg_monthly_transactions,
    }


def _fetch_stored_recommendations(db: Session, customer_id: str, k: int = 3) -> list[dict]:
    rows = (
        db.query(Recommendation)
        .filter(Recommendation.customer_id == customer_id)
        .order_by(Recommendation.score.desc())
        .limit(k)
        .all()
    )
    return [
        {
            "product_id": r.product_id,
            "product": r.product_name,
            "score": r.score,
            "reason_code": r.reason_code,
        }
        for r in rows
    ]


def _handle_spending(db: Session, customer_id: str, query: str) -> dict:
    period = parse_period(query) or default_period()
    summary = compute_spending_summary(db, customer_id, period)
    sentence = render_spending_sentence(summary)
    return {
        "answer": sentence,
        "sources": [],
        "confidence": "high" if summary.has_data else "none",
        "retrieved_chunks": [],
        "pending_action": None,
        "spending_data": summary.to_dict(),
    }


def _handle_which_product(query: str) -> dict:
    result = match_product(query, get_llm_client())
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "confidence": "high" if result["sources"] else "none",
        "retrieved_chunks": [],
        "pending_action": None,
        "spending_data": None,
    }


def _handle_card_action(db: Session, customer_id: str, query: str, action: str) -> dict:
    card = get_customer_card(db, customer_id)
    if card is None:
        return {
            "answer": NO_CARD_ON_FILE,
            "sources": [],
            "confidence": "none",
            "retrieved_chunks": [],
            "pending_action": None,
            "spending_data": None,
        }

    policy = answer_query(query)
    verb = "Freeze" if action == "freeze_card" else "Unfreeze"
    already = (action == "freeze_card" and card.status == "frozen") or (
        action == "unfreeze_card" and card.status == "active"
    )
    if already:
        state = "frozen" if card.status == "frozen" else "active"
        pending = None
        answer = f"Your card ending {card.last4} is already {state}."
    else:
        pending = {
            "action": action,
            "card_id": card.card_id,
            "last4": card.last4,
            "message": f"{verb} your card ending {card.last4}?",
        }
        answer = policy.answer

    return {
        "answer": answer,
        "sources": policy.sources,
        "confidence": policy.confidence,
        "retrieved_chunks": policy.retrieved_chunks,
        "pending_action": pending,
        "spending_data": None,
    }


def handle_recommendation_request(db: Session, customer_id: str, query: str | None) -> dict | None:
    start = time.perf_counter()

    customer = db.get(Customer, customer_id)
    if customer is None:
        return None

    segment = customer.segment_label
    profile = _profile_dict(customer)

    top_k = _fetch_stored_recommendations(db, customer_id, k=3)
    if not top_k:
        # offline batch hasn't run yet for this customer; compute live as a fallback
        top_k = _recommender.score(profile, top_k=3)

    auditor = get_fairness_auditor()
    fairness_flags = auditor.check_against_latest_report(segment, top_k)

    rag_context_found = False
    assistant: dict
    if query:
        intent = detect_intent(query)
        if intent == "spending":
            assistant = _handle_spending(db, customer_id, query)
        elif intent in ("freeze_card", "unfreeze_card"):
            assistant = _handle_card_action(db, customer_id, query, intent)
        elif classify_question_type(query, get_llm_client()) == "which_product":
            assistant = _handle_which_product(query)
        else:
            result = answer_query(query)
            if result.answer == NOT_FOUND_MESSAGE and _MENTIONS_PRODUCT_CATEGORY.search(query):
                # The classifier can miscall a borderline "which product for X"
                # question as "fact", and plain keyword retrieval can miss a
                # doc that doesn't happen to share the customer's exact words.
                # Give the catalog matcher a second try — but only when the
                # question actually names a product category; otherwise a
                # vague or unrelated question could get an LLM to force-fit
                # some product onto it just because it was asked to pick one.
                fallback = _handle_which_product(query)
                assistant = fallback if fallback["sources"] else {
                    "answer": result.answer,
                    "sources": result.sources,
                    "confidence": result.confidence,
                    "retrieved_chunks": result.retrieved_chunks,
                    "pending_action": None,
                    "spending_data": None,
                }
            else:
                assistant = {
                    "answer": result.answer,
                    "sources": result.sources,
                    "confidence": result.confidence,
                    "retrieved_chunks": result.retrieved_chunks,
                    "pending_action": None,
                    "spending_data": None,
                }
        rag_context_found = assistant["confidence"] != "none"
    else:
        assistant = {
            "answer": render_default_summary(profile, top_k),
            "sources": [],
            "confidence": "high" if top_k else "none",
            "retrieved_chunks": [],
            "pending_action": None,
            "spending_data": None,
        }

    response = {
        "customer_id": customer_id,
        "segment": segment,
        "recommendations": [
            {
                "product": r["product"],
                "score": r["score"],
                "reason_code": r["reason_code"],
                "plain_text": reason_code_to_plain_text(r["reason_code"]),
                "learn_more": product_learn_more(r["product"]),
            }
            for r in top_k
        ],
        "assistant_response": assistant,
        "fairness_flags": fairness_flags,
        "timestamp": datetime.now(timezone.utc),
    }

    latency_ms = (time.perf_counter() - start) * 1000
    db.add(
        InteractionLog(
            customer_id=customer_id,
            query=query,
            response_json={**response, "timestamp": response["timestamp"].isoformat()},
            latency_ms=latency_ms,
            rag_context_found=rag_context_found,
        )
    )
    db.commit()

    logger.info(
        "recommendation_request_handled",
        customer_id=customer_id,
        has_query=query is not None,
        latency_ms=round(latency_ms, 2),
        rag_context_found=rag_context_found,
        fairness_flags=len(fairness_flags),
    )

    return response
