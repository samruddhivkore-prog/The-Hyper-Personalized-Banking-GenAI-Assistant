import shutil
import tempfile
from pathlib import Path

from app.models import Customer
from app.services.offline_refresh import run_offline_refresh


def _add_customer(db, **overrides):
    base = dict(
        age=30,
        income_bracket="50k_75k",
        account_tenure_months=24,
        products_held=["checking"],
        channel_preference="mobile",
        avg_monthly_transactions=20,
    )
    base.update(overrides)
    base.setdefault("customer_id", "X")
    customer = Customer(**base)
    db.add(customer)
    return customer


def test_run_offline_refresh_populates_segments_recommendations_and_report(db_session):
    for i in range(12):
        _add_customer(
            db_session,
            customer_id=f"CUST{i:03d}",
            age=25 + i,
            avg_monthly_transactions=10 + i * 3,
        )
    db_session.commit()

    tmp_reports = tempfile.mkdtemp()
    try:
        result = run_offline_refresh(db_session, k=3, fairness_threshold=0.10, reports_dir=tmp_reports)

        assert result["segmentation"]["customers"] == 12
        assert result["recommendations_written"] >= 0

        customers = db_session.query(Customer).all()
        assert all(c.segment_id is not None for c in customers)
        assert all(c.segment_label for c in customers)

        report_files = list(Path(tmp_reports).glob("fairness_*.json"))
        assert len(report_files) == 2  # one per audited dimension: age_bracket, income_bracket

        assert set(result["fairness_report"].keys()) == {"age_bracket", "income_bracket"}
        for dim_report in result["fairness_report"].values():
            assert dim_report["threshold"] == 0.10
    finally:
        shutil.rmtree(tmp_reports, ignore_errors=True)


def test_run_offline_refresh_handles_empty_db(db_session):
    result = run_offline_refresh(db_session, k=5, reports_dir=tempfile.mkdtemp())
    assert result["segmentation"]["customers"] == 0
