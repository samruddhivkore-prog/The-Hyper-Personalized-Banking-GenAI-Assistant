"""Seed the database with synthetic customers, then run the offline refresh
(segmentation + recommendations + fairness audit) so the API has data to
serve immediately after a fresh clone.

Usage: python -m scripts.seed_db [--n 300]
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import SessionLocal, init_db
from app.logging_config import configure_logging, get_logger
from app.models import Customer, Transaction
from app.services.offline_refresh import run_offline_refresh
from scripts.generate_synthetic_customers import generate_customers

logger = get_logger(__name__)


def seed(n: int = 300) -> None:
    configure_logging()
    init_db()
    db = SessionLocal()
    try:
        existing = db.query(Customer).count()
        if existing > 0:
            logger.info("seed_skipped_existing_data", existing_customers=existing)
        else:
            profiles = generate_customers(n)
            for p in profiles:
                customer = Customer(
                    customer_id=p["customer_id"],
                    age=p["age"],
                    income_bracket=p["income_bracket"],
                    account_tenure_months=p["account_tenure_months"],
                    products_held=p["products_held"],
                    channel_preference=p["channel_preference"],
                    avg_monthly_transactions=p["avg_monthly_transactions"],
                )
                db.add(customer)
                for t in p["recent_transactions"]:
                    db.add(
                        Transaction(
                            customer_id=p["customer_id"],
                            date=datetime.strptime(t["date"], "%Y-%m-%d"),
                            category=t["category"],
                            amount=t["amount"],
                        )
                    )
            db.commit()
            logger.info("seed_complete", customers=n)

        result = run_offline_refresh(db)
        logger.info("offline_refresh_after_seed_complete", **result["segmentation"])
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300, help="number of synthetic customers to generate")
    args = parser.parse_args()
    seed(args.n)
