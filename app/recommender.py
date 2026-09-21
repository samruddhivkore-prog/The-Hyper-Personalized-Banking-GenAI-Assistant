"""Next-best-action recommender: rules/scoring engine over the product catalog.

Every recommendation carries a reason_code so the GenAI layer can explain it
truthfully instead of inventing a justification (see app/rag/prompts.py).
"""
from dataclasses import dataclass
from typing import Callable

# (product_id, name, category) -> scoring_fn(profile) -> (score 0..1, reason_code) | None
ScoreFn = Callable[[dict], tuple[float, str] | None]


@dataclass
class Product:
    product_id: str
    name: str
    category: str
    score_fn: ScoreFn


def _txn_freq(profile: dict) -> float:
    return float(profile.get("avg_monthly_transactions") or 0)


def _has_product(profile: dict, product: str) -> bool:
    return product in (profile.get("products_held") or [])


def _income_rank(profile: dict) -> int:
    order = ["under_25k", "25k_50k", "50k_75k", "75k_100k", "100k_plus"]
    try:
        return order.index(profile.get("income_bracket"))
    except ValueError:
        return 2


def score_real_time_payments(profile: dict) -> tuple[float, str] | None:
    freq = _txn_freq(profile)
    if freq < 15:
        return None
    score = min(1.0, 0.5 + (freq - 15) / 60)
    return round(score, 2), "high_transfer_frequency"


def score_high_yield_savings(profile: dict) -> tuple[float, str] | None:
    if _has_product(profile, "high_yield_savings"):
        return None
    if profile.get("account_tenure_months", 0) < 6:
        return None
    tenure_bonus = min(0.3, profile["account_tenure_months"] / 300)
    income_bonus = _income_rank(profile) * 0.08
    score = min(1.0, 0.45 + tenure_bonus + income_bonus)
    return round(score, 2), "stable_balance_growth"


def score_rewards_credit_card(profile: dict) -> tuple[float, str] | None:
    if _has_product(profile, "rewards_credit_card"):
        return None
    if _income_rank(profile) < 1:
        return None
    freq = _txn_freq(profile)
    if freq < 10:
        return None
    score = min(1.0, 0.4 + freq / 100 + _income_rank(profile) * 0.05)
    return round(score, 2), "frequent_spender_eligible_income"


def score_personal_loan(profile: dict) -> tuple[float, str] | None:
    if _has_product(profile, "personal_loan"):
        return None
    if profile.get("account_tenure_months", 0) < 12:
        return None
    if _income_rank(profile) < 1:
        return None
    score = min(1.0, 0.35 + profile["account_tenure_months"] / 400 + _income_rank(profile) * 0.06)
    return round(score, 2), "established_tenure_eligible_income"


def score_mobile_check_deposit(profile: dict) -> tuple[float, str] | None:
    if profile.get("channel_preference") not in ("mobile", "web"):
        return None
    if _has_product(profile, "mobile_check_deposit"):
        return None
    return 0.55, "digital_channel_preference"


def score_overdraft_protection(profile: dict) -> tuple[float, str] | None:
    if _has_product(profile, "overdraft_protection"):
        return None
    freq = _txn_freq(profile)
    if freq < 25:
        return None
    return round(min(0.9, 0.4 + freq / 80), 2), "high_txn_volume_overdraft_risk"


def score_student_loan_refinance(profile: dict) -> tuple[float, str] | None:
    if profile.get("age", 99) > 35:
        return None
    if _has_product(profile, "student_loan_refinance"):
        return None
    return 0.5, "young_customer_debt_optimization"


def score_retirement_ira(profile: dict) -> tuple[float, str] | None:
    if profile.get("age", 0) < 30:
        return None
    if _has_product(profile, "retirement_ira"):
        return None
    score = min(1.0, 0.4 + (profile["age"] - 30) / 100 + _income_rank(profile) * 0.05)
    return round(score, 2), "long_term_wealth_building_age_appropriate"


CATALOG: list[Product] = [
    Product("real_time_payments", "Real-Time Payments", "payments", score_real_time_payments),
    Product("high_yield_savings", "High-Yield Savings", "savings", score_high_yield_savings),
    Product("rewards_credit_card", "Rewards Credit Card", "credit", score_rewards_credit_card),
    Product("personal_loan", "Personal Loan", "lending", score_personal_loan),
    Product("mobile_check_deposit", "Mobile Check Deposit", "servicing", score_mobile_check_deposit),
    Product("overdraft_protection", "Overdraft Protection", "servicing", score_overdraft_protection),
    Product("student_loan_refinance", "Student Loan Refinance", "lending", score_student_loan_refinance),
    Product("retirement_ira", "Retirement IRA", "investing", score_retirement_ira),
]


class RecommenderModel:
    def __init__(self, catalog: list[Product] | None = None):
        self.catalog = catalog or CATALOG

    def score(self, profile: dict, top_k: int | None = None) -> list[dict]:
        candidates = []
        for product in self.catalog:
            result = product.score_fn(profile)
            if result is None:
                continue
            score, reason_code = result
            candidates.append(
                {
                    "product_id": product.product_id,
                    "product": product.name,
                    "score": score,
                    "reason_code": reason_code,
                }
            )
        candidates.sort(key=lambda c: c["score"], reverse=True)
        if top_k is not None:
            candidates = candidates[:top_k]
        return candidates
