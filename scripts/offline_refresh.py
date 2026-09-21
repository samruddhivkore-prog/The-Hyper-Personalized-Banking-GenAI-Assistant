"""CLI entrypoint for the offline/batch pipeline (spec section 8.1).
Intended to run on a schedule (cron, GitHub Actions scheduled workflow, etc).

Usage: python -m scripts.offline_refresh
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.database import SessionLocal, init_db
from app.logging_config import configure_logging, get_logger
from app.services.offline_refresh import run_offline_refresh
from scripts.index_kb import index_kb

logger = get_logger(__name__)


def main() -> None:
    configure_logging()
    init_db()
    settings = get_settings()

    index_kb()

    db = SessionLocal()
    try:
        result = run_offline_refresh(
            db, k=settings.segmentation_k, fairness_threshold=settings.fairness_parity_threshold
        )
        logger.info("offline_refresh_complete", **result["segmentation"])
    finally:
        db.close()


if __name__ == "__main__":
    main()
