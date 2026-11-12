"""Question-type classification for /chat, beyond the deterministic
spending/freeze/unfreeze routing in app/services/intent.py (which stays
rule-based for safety — see that module's docstring). Everything else (fact
lookup, how-to steps, or "which product fits my purpose") genuinely needs
judgement, so a real LLM provider classifies it — but only ever returns one
label from a fixed set, which is validated here before use; an unrecognised
or malformed label always falls back to "fact", never trusted blindly."""
import json
import re

from app.rag.llm_client import LLMClient, StubLLMClient
from app.services.intent import is_how_to

QUESTION_TYPES = {"fact", "howto", "which_product", "spending", "action", "other"}

CLASSIFY_PROMPT = (
    "Classify the customer's banking question into exactly one label: "
    "fact, howto, which_product, spending, action, or other.\n"
    "- fact: asks for a specific detail about a named product (a fee, rate, limit, speed).\n"
    "- howto: asks for the steps to do something (e.g. 'how do I ...').\n"
    "- which_product: asks which product fits a purpose, without naming a specific product "
    "(e.g. 'which loan for a car', 'what account is best for saving').\n"
    "- spending: asks about the customer's own spending or transaction history.\n"
    "- action: asks to freeze, unfreeze, or otherwise change something on the account.\n"
    "- other: anything else.\n"
    'Reply with ONLY a JSON object like {"type": "fact"}, nothing else.\n\n'
    "Question: {query}"
)


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _classify_rule_based(query: str) -> str:
    """No-LLM fallback: coarse but safe — never anything but a valid label."""
    if is_how_to(query):
        return "howto"
    if re.search(r"\bwhich\b|\bwhat.*best\b|\bshould i (get|take|use)\b", query, re.IGNORECASE):
        return "which_product"
    return "fact"


def classify_question_type(query: str, llm_client: LLMClient) -> str:
    if isinstance(llm_client, StubLLMClient):
        return _classify_rule_based(query)

    try:
        raw = llm_client.generate(CLASSIFY_PROMPT.format(query=query), context_chunks=[], temperature=0.0)
    except Exception:
        return _classify_rule_based(query)

    label = str(_extract_json(raw).get("type", "")).strip().lower()
    return label if label in QUESTION_TYPES else "fact"
