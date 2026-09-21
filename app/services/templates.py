"""Default (no-query) summary template — used when a request omits `query`."""


def render_default_summary(profile: dict, recommendations: list[dict]) -> str:
    name_bit = f"customer {profile.get('customer_id', '')}".strip()
    if not recommendations:
        return f"Hi {name_bit}, we don't have any new personalized recommendations for you right now."

    top = recommendations[0]
    lines = [f"Hi {name_bit}, here are your top personalized recommendations:"]
    for rec in recommendations:
        lines.append(f"- {rec['product']} (reason: {rec['reason_code'].replace('_', ' ')})")
    lines.append(
        f"\nWe think {top['product']} is your best next step based on your recent activity."
    )
    return "\n".join(lines)
