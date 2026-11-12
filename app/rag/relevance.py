"""Relevance and fact-extraction helpers shared by the pipeline and the stub
LLM. Embedding cosine scores alone aren't a reliable gate — they're
calibrated differently across backends (the hash-fallback embedding used
when sentence-transformers/chromadb aren't available produces much lower
absolute scores than real embeddings, even for genuinely relevant matches).
Keyword overlap is backend-agnostic and is the primary relevance signal;
embedding score is only used to rank among keyword-passing candidates.
"""
import re

STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "for", "and", "or", "what",
    "how", "do", "i", "my", "in", "on", "with", "can", "does", "it", "this",
    "that", "you", "your", "be", "was", "were", "fastest", "way", "offer",
    "get", "have", "has", "want", "need", "like", "please", "would",
}

TIME_WORDS = {"second", "seconds", "day", "days", "hour", "hours", "minute", "minutes", "business"}
FEE_WORDS = {"fee", "fees", "cost", "costs", "charge", "charges", "free"}


def _stem(word: str) -> str:
    """Minimal suffix stripping so "sending"/"sends"/"sent" and "loan"/"loans"
    are treated as the same keyword. Exact-token matching otherwise misses
    genuinely relevant sentences just because a document uses a different
    inflection than the question did (e.g. a policy doc saying "the sending
    account" doesn't exact-match a question asking "how can I send money")."""
    if len(word) > 5 and word.endswith("ing"):
        return word[:-3]
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith("es"):
        return word[:-2]
    if len(word) > 4 and word.endswith("ed"):
        return word[:-2]
    if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def stemmed_tokens(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [s for s in (_stem(t) for t in tokens if len(t) > 2) if s not in STOPWORDS]


def keywords(text: str) -> set[str]:
    return set(stemmed_tokens(text))


def fact_score(sentence: str) -> int:
    """How much concrete, citable fact (fee, limit, time) a sentence carries —
    used to prefer factual sentences over vague connective ones when building
    the extractive fallback answer. Numeric tokens are counted, not just
    detected, so e.g. "range from €1,000 to €50,000, with terms of 12, 24,
    36... months" (several distinct figures) clearly outranks a sentence
    that merely mentions one unrelated number in passing — without this, two
    sentences that both "have a number" tie, and the tie-break falls back to
    document order, which can surface the wrong figure entirely (an income
    eligibility threshold instead of the loan amount actually asked about).
    A sentence pairing € with an actual fee/cost word is boosted well above
    one that merely mentions a limit, so e.g. "costs a flat fee of €1.00"
    outranks "you can send up to €2,500" when both are candidates."""
    words = set(re.findall(r"[a-z]+", sentence.lower()))
    score = sentence.count("€")
    score += min(len(re.findall(r"\d[\d,]*", sentence)), 4)
    if "€" in sentence and (words & FEE_WORDS):
        score += 4
    if words & TIME_WORDS:
        score += 2
    return score
