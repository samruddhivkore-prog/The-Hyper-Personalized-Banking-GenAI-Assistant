"""Online pipeline: HandleRecommendationRequest (spec section 8.2)."""
import time
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from app.fairness import get_fairness_auditor
from app.logging_config import get_logger
from app.models import Customer, InteractionLog, Recommendation
from app.rag.pipeline import answer_query
from app.recommender import RecommenderModel
from app.services.templates import render_default_summary

logger = get_logger(__name__)

_recommender = RecommenderModel()


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
    if query:
        result = answer_query(query, segment, top_k)
        answer, sources, confidence = result.answer, result.sources, result.confidence
        rag_context_found = confidence != "none"
    else:
        answer = render_default_summary(profile, top_k)
        sources = []
        confidence = "high" if top_k else "none"

    response = {
        "customer_id": customer_id,
        "segment": segment,
        "recommendations": [
            {"product": r["product"], "score": r["score"], "reason_code": r["reason_code"]}
            for r in top_k
        ],
        "assistant_response": {"answer": answer, "sources": sources, "confidence": confidence},
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
