"""Rule-based chat intent routing. Deliberately not LLM-based: which code
path handles a message (spending lookup vs. a database-mutating card action
vs. policy Q&A) must be deterministic, not a model's guess."""
import re

SPENDING_WORDS = re.compile(r"\b(spend|spent|spending)\b", re.IGNORECASE)
FREEZE_WORDS = re.compile(r"\bfreeze\b", re.IGNORECASE)
UNFREEZE_WORDS = re.compile(r"\b(unfreeze|un-freeze|reactivate)\b", re.IGNORECASE)
HOW_TO_WORDS = re.compile(r"\bhow (do|can|to) i\b|\bhow to\b", re.IGNORECASE)


def detect_intent(query: str) -> str:
    if UNFREEZE_WORDS.search(query):
        return "unfreeze_card"
    if FREEZE_WORDS.search(query):
        return "freeze_card"
    if SPENDING_WORDS.search(query):
        return "spending"
    return "policy"


def is_how_to(query: str) -> bool:
    """"How do I send money" asks for the procedure — numbered steps.
    "How can I send 1000 to my dad" names a specific amount and is really
    asking what happens (fee, speed, limit) for that transfer, not for the
    click-by-click procedure, even though it's phrased the same way — so a
    query naming a specific figure is excluded from step-routing."""
    if not HOW_TO_WORDS.search(query):
        return False
    return not re.search(r"\d", query)
