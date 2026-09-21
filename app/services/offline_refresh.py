"""Offline / batch pipeline (spec section 8.1): OfflineRefresh().

Recomputes segments, refreshes recommendations, runs the fairness audit, and
writes a dated fairness report artifact. Run on a schedule, not per-request —
see scripts/offline_refresh.py for the CLI entrypoint.
"""
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.fairness import FairnessAuditor, age_bracket
from app.logging_config import get_logger
from app.models import Customer, Recommendation
from app.recommender import RecommenderModel
from app.segmentation import SegmentationModel

logger = get_logger(__name__)


def _customer_to_profile(customer: Customer) -> dict:
    return {
        "customer_id": customer.customer_id,
        "age": customer.age,
        "income_bracket": customer.income_bracket,
        "account_tenure_months": customer.account_tenure_months,
        "products_held": customer.products_held or [],
        "channel_preference": customer.channel_preference,
        "avg_monthly_transactions": customer.avg_monthly_transactions,
    }


def refresh_segments(db: Session, k: int = 5) -> dict:
    customers = db.query(Customer).all()
    if not customers:
        return {"customers": 0, "silhouette": None}

    profiles = [_customer_to_profile(c) for c in customers]
    model = SegmentationModel(k=k)
    result = model.fit(profiles)

    for customer in customers:
        seg_id = result.assignments[customer.customer_id]
        customer.segment_id = seg_id
        customer.segment_label = result.labels[seg_id]
    db.commit()

    logger.info("segments_refreshed", customers=len(customers), silhouette=result.silhouette)
    return {"customers": len(customers), "silhouette": result.silhouette}


def refresh_recommendations(db: Session, top_k: int = 5) -> int:
    customers = db.query(Customer).all()
    recommender = RecommenderModel()
    total = 0

    for customer in customers:
        db.query(Recommendation).filter(Recommendation.customer_id == customer.customer_id).delete()
        profile = _customer_to_profile(customer)
        candidates = recommender.score(profile, top_k=top_k)
        for c in candidates:
            db.add(
                Recommendation(
                    customer_id=customer.customer_id,
                    product_id=c["product_id"],
                    product_name=c["product"],
                    score=c["score"],
                    reason_code=c["reason_code"],
                    segment_id=customer.segment_id,
                )
            )
            total += 1
    db.commit()

    logger.info("recommendations_refreshed", customers=len(customers), recommendations=total)
    return total


def _group_key(customer: Customer, dimension: str) -> str:
    if dimension == "age_bracket":
        return age_bracket(customer.age)
    if dimension == "income_bracket":
        return customer.income_bracket or "unknown"
    return customer.segment_label or "unassigned"


def run_fairness_audit(db: Session, threshold: float, window_hours: int = 24, reports_dir: str = "reports"):
    """Audits recommendation parity across protected-adjacent attributes
    (age bracket, income bracket) — not the KMeans segment, which is itself
    derived from these attributes and would make every product look unfair
    purely as a clustering artifact. See app/fairness.py module docstring."""
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    recent = db.query(Recommendation).filter(Recommendation.created_at >= cutoff).all()
    customers = db.query(Customer).all()
    customer_lookup = {c.customer_id: c for c in customers}

    auditor = FairnessAuditor(threshold=threshold)
    date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
    reports = {}

    for dimension in ("age_bracket", "income_bracket"):
        group_sizes: dict[str, int] = {}
        for c in customers:
            key = _group_key(c, dimension)
            group_sizes[key] = group_sizes.get(key, 0) + 1

        recs_for_audit = []
        for r in recent:
            customer = customer_lookup.get(r.customer_id)
            group = _group_key(customer, dimension) if customer else "unassigned"
            recs_for_audit.append({"product_id": r.product_id, "group": group})

        report = auditor.compute_parity(recs_for_audit, group_sizes, group_by=dimension)
        path = f"{reports_dir}/fairness_{dimension}_{date_str}.json"
        report.save(path)
        reports[dimension] = report

        logger.info(
            "fairness_audit_complete",
            dimension=dimension,
            flagged_products=len(report.flagged_product_ids()),
            unjustified_flags=len(report.flagged_unjustified_product_ids()),
            report_path=path,
        )

    return reports


def run_offline_refresh(
    db: Session, k: int = 5, fairness_threshold: float = 0.10, reports_dir: str = "reports"
) -> dict:
    seg_result = refresh_segments(db, k=k)
    rec_count = refresh_recommendations(db)
    reports = run_fairness_audit(db, threshold=fairness_threshold, reports_dir=reports_dir)
    return {
        "segmentation": seg_result,
        "recommendations_written": rec_count,
        "fairness_report": {dim: r.to_dict() for dim, r in reports.items()},
    }
