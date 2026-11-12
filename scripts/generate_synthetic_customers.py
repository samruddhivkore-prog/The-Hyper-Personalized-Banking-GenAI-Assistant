"""Synthetic customer profile generator.

Generates realistic-but-fake customer profiles matching the schema in spec
section 5.2, in the spirit of IBM/TabFormer-style synthetic transaction data —
no real customer data is used anywhere in this project.
"""
import random
from datetime import datetime, timedelta, timezone

INCOME_BRACKETS = ["under_25k", "25k_50k", "50k_75k", "75k_100k", "100k_plus"]
CHANNELS = ["mobile", "web", "branch", "phone"]
PRODUCTS = [
    "checking",
    "savings",
    "high_yield_savings",
    "rewards_credit_card",
    "personal_loan",
    "mobile_cheque_deposit",
    "overdraft_protection",
    "student_loan_refinance",
    "pension_prsa",
]
SPENDING_CATEGORIES = ["groceries", "transport", "dining", "bills", "shopping", "other"]
_CATEGORY_WEIGHTS = [0.30, 0.15, 0.15, 0.20, 0.15, 0.05]
_CATEGORY_AMOUNT_RANGE = {
    "groceries": (15, 120),
    "transport": (5, 60),
    "dining": (10, 90),
    "bills": (30, 220),
    "shopping": (10, 200),
    "other": (5, 80),
}

# Weighted so segments emerge naturally from correlated attributes rather than
# being forced, e.g. younger + digital-first customers transact more often.
_INCOME_WEIGHTS = [0.15, 0.30, 0.25, 0.18, 0.12]


def _pick_income(age: int) -> str:
    if age < 28:
        weights = [0.30, 0.35, 0.20, 0.10, 0.05]
    elif age < 45:
        weights = [0.10, 0.25, 0.30, 0.20, 0.15]
    else:
        weights = [0.08, 0.20, 0.27, 0.25, 0.20]
    return random.choices(INCOME_BRACKETS, weights=weights)[0]


def _generate_transactions(monthly_count: int, months_back: int = 12) -> list[dict]:
    """Spreads roughly `monthly_count` transactions per month over the last
    `months_back` months, so a spending lookup for any recent month —
    not just the most recent 30 days — has real data behind it."""
    today = datetime.now(timezone.utc)
    txns = []
    per_month = max(2, min(int(monthly_count), 15))
    for month_offset in range(months_back):
        for _ in range(per_month):
            days_ago = month_offset * 30 + random.randint(0, 29)
            category = random.choices(SPENDING_CATEGORIES, weights=_CATEGORY_WEIGHTS)[0]
            lo, hi = _CATEGORY_AMOUNT_RANGE[category]
            txns.append(
                {
                    "date": (today - timedelta(days=days_ago)).strftime("%Y-%m-%d"),
                    "category": category,
                    "amount": round(random.uniform(lo, hi), 2),
                }
            )
    return txns


def generate_customer(index: int) -> dict:
    customer_id = f"CUST{index:05d}"
    age = random.randint(18, 75)
    income_bracket = _pick_income(age)
    account_tenure_months = random.randint(1, 240)
    channel_preference = random.choices(CHANNELS, weights=[0.45, 0.25, 0.20, 0.10])[0]

    # digital-first + younger customers transact more frequently, on average
    base_freq = 8
    if channel_preference in ("mobile", "web"):
        base_freq += 12
    if age < 35:
        base_freq += 10
    avg_monthly_transactions = max(1, round(random.gauss(base_freq, 8), 1))

    num_products = random.choices([1, 2, 3, 4], weights=[0.35, 0.35, 0.20, 0.10])[0]
    products_held = random.sample(PRODUCTS, k=min(num_products, len(PRODUCTS)))
    if "checking" not in products_held:
        products_held.insert(0, "checking")

    return {
        "customer_id": customer_id,
        "age": age,
        "income_bracket": income_bracket,
        "account_tenure_months": account_tenure_months,
        "products_held": products_held,
        "channel_preference": channel_preference,
        "avg_monthly_transactions": avg_monthly_transactions,
        "recent_transactions": _generate_transactions(int(avg_monthly_transactions)),
    }


def generate_card(customer_id: str) -> dict:
    last4 = f"{random.randint(0, 9999):04d}"
    return {"card_id": f"CARD-{customer_id}", "customer_id": customer_id, "last4": last4, "status": "active"}


def generate_customers(n: int, seed: int | None = 42) -> list[dict]:
    if seed is not None:
        random.seed(seed)
    return [generate_customer(i + 1) for i in range(n)]
