"""Default (no-query) summary template — used when a request omits `query`.

Deliberately does NOT list products or name a top pick: recommendations are
shown once, as cards, by the frontend. Repeating them here was the actual
cause of "recommendations appear twice" — see app/services/recommendation_service.py.
"""


def render_default_summary(profile: dict, recommendations: list[dict]) -> str:
    if not recommendations:
        return "We don't have any new recommendations for you right now."
    return "Here are a few things we think could help, based on your account."
