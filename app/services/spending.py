"""Spending summary: every figure here comes straight from the transactions
table. No LLM is involved in computing a total, a percentage, or a
comparison — only (optionally) in recognising which period a free-text
question refers to, and that period is parsed and validated by
app/services/period_parser.py before this module ever sees it."""
from dataclasses import dataclass, field
from datetime import datetime, time

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.models import Transaction
from app.services.period_parser import Period, previous_period

CATEGORY_ORDER = ["groceries", "transport", "dining", "bills", "shopping", "other"]
NO_DATA_MESSAGE = "No transactions found for that period."


@dataclass
class CategoryBreakdown:
    category: str
    amount: float
    pct: float


@dataclass
class SpendingSummary:
    period_label: str
    start: str
    end: str
    total: float
    categories: list[CategoryBreakdown] = field(default_factory=list)
    top_categories: list[str] = field(default_factory=list)
    previous_total: float | None = None
    change_pct: float | None = None
    has_data: bool = True

    def to_dict(self) -> dict:
        return {
            "period_label": self.period_label,
            "start": self.start,
            "end": self.end,
            "total": round(self.total, 2),
            "categories": [
                {"category": c.category, "amount": round(c.amount, 2), "pct": round(c.pct, 1)}
                for c in self.categories
            ],
            "top_categories": self.top_categories,
            "previous_total": round(self.previous_total, 2) if self.previous_total is not None else None,
            "change_pct": round(self.change_pct, 1) if self.change_pct is not None else None,
            "has_data": self.has_data,
        }


def _total_for_range(db: Session, customer_id: str, period: Period) -> float:
    start_dt = datetime.combine(period.start, time.min)
    end_dt = datetime.combine(period.end, time.max)
    result = (
        db.query(func.sum(Transaction.amount))
        .filter(
            Transaction.customer_id == customer_id,
            Transaction.date >= start_dt,
            Transaction.date <= end_dt,
        )
        .scalar()
    )
    return float(result or 0.0)


def compute_spending_summary(db: Session, customer_id: str, period: Period) -> SpendingSummary:
    start_dt = datetime.combine(period.start, time.min)
    end_dt = datetime.combine(period.end, time.max)

    rows = (
        db.query(Transaction.category, func.sum(Transaction.amount))
        .filter(
            Transaction.customer_id == customer_id,
            Transaction.date >= start_dt,
            Transaction.date <= end_dt,
        )
        .group_by(Transaction.category)
        .all()
    )

    total = sum(amount for _, amount in rows)
    if total <= 0:
        return SpendingSummary(
            period_label=period.label,
            start=period.start.isoformat(),
            end=period.end.isoformat(),
            total=0.0,
            has_data=False,
        )

    categories = sorted(
        (CategoryBreakdown(cat, float(amt), float(amt) / total * 100) for cat, amt in rows),
        key=lambda c: c.amount,
        reverse=True,
    )
    top_categories = [c.category for c in categories[:3]]

    previous_total = _total_for_range(db, customer_id, previous_period(period))
    change_pct = None
    if previous_total > 0:
        change_pct = (total - previous_total) / previous_total * 100

    return SpendingSummary(
        period_label=period.label,
        start=period.start.isoformat(),
        end=period.end.isoformat(),
        total=total,
        categories=categories,
        top_categories=top_categories,
        previous_total=previous_total,
        change_pct=change_pct,
    )


def render_spending_sentence(summary: SpendingSummary) -> str:
    if not summary.has_data:
        return NO_DATA_MESSAGE

    top = summary.categories[0]
    start_dt = datetime.fromisoformat(summary.start)
    end_dt = datetime.fromisoformat(summary.end)
    start_label = f"{start_dt.day} {start_dt.strftime('%b')}"
    end_label = f"{end_dt.day} {end_dt.strftime('%b')}"
    sentence = (
        f"You spent €{summary.total:,.0f} between {start_label} and {end_label}. "
        f"Most went on {top.category} (€{top.amount:,.0f})."
    )
    if summary.change_pct is not None:
        direction = "more" if summary.change_pct > 0 else "less"
        sentence += f" That is {abs(summary.change_pct):.0f}% {direction} than the period before."
    return sentence
