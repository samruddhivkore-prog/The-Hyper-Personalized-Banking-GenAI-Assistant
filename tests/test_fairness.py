import pytest

from app.fairness import FairnessAuditor


def test_no_flag_when_rates_are_within_threshold():
    auditor = FairnessAuditor(threshold=0.10)
    recs = [
        {"product_id": "high_yield_savings", "group": "segment_a"},
        {"product_id": "high_yield_savings", "group": "segment_b"},
    ]
    group_sizes = {"segment_a": 10, "segment_b": 10}
    report = auditor.compute_parity(recs, group_sizes)

    product = next(p for p in report.products if p.product_id == "high_yield_savings")
    assert product.parity_diff == 0.0
    assert not product.flagged
    assert report.flagged_product_ids() == set()


def test_flags_product_when_parity_diff_exceeds_threshold():
    auditor = FairnessAuditor(threshold=0.10)
    # segment_a: 8/10 recommended (0.8) vs segment_b: 1/10 (0.1) -> diff 0.7
    recs = [{"product_id": "personal_loan", "group": "segment_a"} for _ in range(8)]
    recs += [{"product_id": "personal_loan", "group": "segment_b"} for _ in range(1)]
    group_sizes = {"segment_a": 10, "segment_b": 10}

    report = auditor.compute_parity(recs, group_sizes)
    product = next(p for p in report.products if p.product_id == "personal_loan")

    assert product.flagged
    assert product.parity_diff == pytest.approx(0.7)
    assert "personal_loan" in report.flagged_product_ids()


def test_intentionally_biased_case_is_caught():
    """A deliberately biased scenario, to verify the auditor actually catches
    unfair treatment rather than always passing."""
    auditor = FairnessAuditor(threshold=0.10)
    recs = []
    # affluent segment recommended to nearly everyone, low-income segment almost never
    recs += [{"product_id": "rewards_credit_card", "group": "affluent"} for _ in range(19)]
    recs += [{"product_id": "rewards_credit_card", "group": "low_income"} for _ in range(1)]
    group_sizes = {"affluent": 20, "low_income": 20}

    report = auditor.compute_parity(recs, group_sizes)
    assert report.flagged_product_ids() == {"rewards_credit_card"}


def test_check_against_latest_report_returns_human_readable_flags():
    auditor = FairnessAuditor(threshold=0.10)
    recs = [{"product_id": "personal_loan", "group": "segment_a"} for _ in range(9)]
    recs += [{"product_id": "personal_loan", "group": "segment_b"} for _ in range(0)]
    group_sizes = {"segment_a": 10, "segment_b": 10}
    auditor.compute_parity(recs, group_sizes)

    flags = auditor.check_against_latest_report(
        "segment_a", [{"product_id": "personal_loan", "product": "Personal Loan"}]
    )
    assert len(flags) == 1
    assert "personal_loan" in flags[0]


def test_check_against_latest_report_empty_when_no_report_yet():
    auditor = FairnessAuditor(threshold=0.10)
    flags = auditor.check_against_latest_report("segment_a", [{"product_id": "personal_loan"}])
    assert flags == []


def test_report_round_trips_through_dict():
    from app.fairness import FairnessReport

    auditor = FairnessAuditor(threshold=0.10)
    recs = [{"product_id": "personal_loan", "group": "segment_a"}]
    report = auditor.compute_parity(recs, {"segment_a": 10})

    restored = FairnessReport.from_dict(report.to_dict())
    assert restored.threshold == report.threshold
    assert restored.flagged_product_ids() == report.flagged_product_ids()
