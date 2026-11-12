"""Customer-facing copy: plain-English text for things the backend otherwise
expresses as codes, scores, or internal labels. Nothing here should ever leak
a reason_code, a score, or a segment name back to a customer — see
app/recommender.py for the reason codes this maps, one per scoring function.
"""

REASON_CODE_TO_PLAIN_TEXT: dict[str, str] = {
    "high_transfer_frequency": "You send money often, so this could save you time.",
    "stable_balance_growth": "You've kept a steady balance, so this could help it grow further.",
    "frequent_spender_eligible_income": "Based on your spending and income, you could earn more back.",
    "established_tenure_eligible_income": (
        "You've banked with us for a while and your income qualifies."
    ),
    "digital_channel_preference": "Since you mostly bank on mobile, this fits how you already use the app.",
    "high_txn_volume_overdraft_risk": "With how often you transact, this helps avoid surprise declines.",
    "young_customer_debt_optimization": "This could lower what you pay on existing student debt.",
    "long_term_wealth_building_age_appropriate": "This is a good time to start building long-term savings.",
}

_FALLBACK_PLAIN_TEXT = "This could be a good fit based on your account activity."


def reason_code_to_plain_text(reason_code: str) -> str:
    return REASON_CODE_TO_PLAIN_TEXT.get(reason_code, _FALLBACK_PLAIN_TEXT)


PRODUCT_LEARN_MORE: dict[str, str] = {
    "Real-Time Payments": "Send money to another account in seconds, any day, for a small fee.",
    "High-Yield Savings": "A savings account that earns more interest than a standard one, with no fees.",
    "Rewards Credit Card": "Earn cash back on everyday purchases like groceries and dining.",
    "Personal Loan": "A fixed-rate loan for things like debt consolidation or a big purchase.",
    "Mobile Cheque Deposit": "Deposit a cheque by taking a photo in the app — no branch visit needed.",
    "Overdraft Protection": "Automatically covers a payment that would otherwise be declined.",
    "Student Loan Refinance": "Combine your student loans into one, potentially at a lower rate.",
    "Pension (PRSA)": "A tax-advantaged account for saving toward retirement.",
}

_FALLBACK_LEARN_MORE = "Ask us in chat for more details about this product."


def product_learn_more(product_name: str) -> str:
    return PRODUCT_LEARN_MORE.get(product_name, _FALLBACK_LEARN_MORE)
