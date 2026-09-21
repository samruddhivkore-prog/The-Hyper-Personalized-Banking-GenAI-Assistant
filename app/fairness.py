"""Fairness auditor: demographic parity checks over recommendation batches.

Groups by protected-adjacent attributes (age bracket, income bracket) rather
than the KMeans customer segment: segment membership is itself derived from
age/tenure/channel/income, so grouping by segment would flag nearly every
product purely as an artifact of clustering, not because of unfair treatment.

Some parity differences are *expected* and documented: several products have
an explicit eligibility gate on age or income (see JUSTIFIED_PRODUCTS below).
Those are still recorded and flagged in the report for transparency, but are
excluded from the per-request `fairness_flags` surfaced to callers, which is
reserved for differences that aren't explained by a known eligibility rule
and therefore warrant human review.
"""
import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

# product_id -> the attribute its scoring function explicitly gates on, by
# dimension. See app/recommender.py: score_retirement_ira (age >= 30),
# score_student_loan_refinance (age <= 35), score_personal_loan and
# score_rewards_credit_card (income_rank >= 1).
JUSTIFIED_PRODUCTS: dict[str, set[str]] = {
    "age_bracket": {"retirement_ira", "student_loan_refinance"},
    "income_bracket": {"personal_loan", "rewards_credit_card"},
}


def age_bracket(age: int) -> str:
    if age < 30:
        return "under_30"
    if age < 45:
        return "30_44"
    if age < 60:
        return "45_59"
    return "60_plus"


@dataclass
class ProductParity:
    product_id: str
    rates: dict[str, float]
    parity_diff: float
    flagged: bool
    justified: bool = False


@dataclass
class FairnessReport:
    group_by: str
    threshold: float
    products: list[ProductParity]
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def flagged_product_ids(self) -> set[str]:
        return {p.product_id for p in self.products if p.flagged}

    def flagged_unjustified_product_ids(self) -> set[str]:
        return {p.product_id for p in self.products if p.flagged and not p.justified}

    def to_dict(self) -> dict:
        return {
            "group_by": self.group_by,
            "threshold": self.threshold,
            "generated_at": self.generated_at,
            "products": [
                {
                    "product_id": p.product_id,
                    "rates": p.rates,
                    "parity_diff": round(p.parity_diff, 4),
                    "flagged": p.flagged,
                    "justified": p.justified,
                }
                for p in self.products
            ],
        }

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: dict) -> "FairnessReport":
        products = [
            ProductParity(
                product_id=p["product_id"],
                rates=p["rates"],
                parity_diff=p["parity_diff"],
                flagged=p["flagged"],
                justified=p.get("justified", False),
            )
            for p in data["products"]
        ]
        return cls(
            group_by=data["group_by"],
            threshold=data["threshold"],
            products=products,
            generated_at=data["generated_at"],
        )


def load_latest_reports(reports_dir: str = "reports") -> list[FairnessReport]:
    """Loads the most recently written fairness_<dimension>_*.json artifact
    for each dimension found under reports_dir."""
    pattern = os.path.join(reports_dir, "fairness_*.json")
    files = sorted(glob.glob(pattern))
    latest_by_dimension: dict[str, str] = {}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        # later files (sorted by name, which embeds the date) overwrite earlier ones
        latest_by_dimension[data["group_by"]] = path

    reports = []
    for path in latest_by_dimension.values():
        with open(path) as f:
            reports.append(FairnessReport.from_dict(json.load(f)))
    return reports


class FairnessAuditor:
    def __init__(self, threshold: float = 0.10):
        self.threshold = threshold
        self._latest_reports: dict[str, FairnessReport] = {}

    def compute_parity(
        self,
        recommendations: list[dict[str, Any]],
        group_sizes: dict[str, int],
        group_by: str = "segment",
    ) -> FairnessReport:
        """
        recommendations: list of {"product_id": str, "group": str, ...}
        group_sizes: total customer count per group (denominator for selection rate)
        """
        product_ids = sorted({r["product_id"] for r in recommendations})
        justified_ids = JUSTIFIED_PRODUCTS.get(group_by, set())
        products: list[ProductParity] = []

        for product_id in product_ids:
            rates: dict[str, float] = {}
            for group, size in group_sizes.items():
                if size <= 0:
                    continue
                count = sum(
                    1 for r in recommendations if r["product_id"] == product_id and r["group"] == group
                )
                rates[group] = count / size

            if len(rates) < 2:
                parity_diff = 0.0
            else:
                parity_diff = max(rates.values()) - min(rates.values())

            products.append(
                ProductParity(
                    product_id=product_id,
                    rates=rates,
                    parity_diff=parity_diff,
                    flagged=parity_diff > self.threshold,
                    justified=product_id in justified_ids,
                )
            )

        report = FairnessReport(group_by=group_by, threshold=self.threshold, products=products)
        self._latest_reports[group_by] = report
        return report

    def check_against_latest_report(
        self, segment: str | None, recommendations: list[dict[str, Any]]
    ) -> list[str]:
        """Per-request check: does this response include any product with an
        unexplained demographic parity flag (i.e. not covered by a known
        eligibility rule) in any audited dimension? Returns human-readable
        flag strings."""
        flags = []
        for report in self._latest_reports.values():
            unjustified = report.flagged_unjustified_product_ids()
            for rec in recommendations:
                pid = rec.get("product_id") or rec.get("product")
                if pid in unjustified:
                    diff = next(p.parity_diff for p in report.products if p.product_id == pid)
                    flags.append(
                        f"{pid}: demographic parity difference {diff:.2f} exceeds "
                        f"{report.threshold:.2f} threshold across {report.group_by}s"
                    )
        return flags

    def load_report(self, report: FairnessReport) -> None:
        self._latest_reports[report.group_by] = report

    @property
    def latest_reports(self) -> dict[str, FairnessReport]:
        return self._latest_reports

    @property
    def latest_report(self) -> FairnessReport | None:
        """Convenience accessor for the single-dimension case (tests, simple callers)."""
        if not self._latest_reports:
            return None
        return next(iter(self._latest_reports.values()))


_auditor_singleton: FairnessAuditor | None = None


def get_fairness_auditor(threshold: float | None = None) -> FairnessAuditor:
    global _auditor_singleton
    if _auditor_singleton is None:
        from app.config import get_settings

        _auditor_singleton = FairnessAuditor(
            threshold=threshold or get_settings().fairness_parity_threshold
        )
        for report in load_latest_reports():
            _auditor_singleton.load_report(report)
    return _auditor_singleton
