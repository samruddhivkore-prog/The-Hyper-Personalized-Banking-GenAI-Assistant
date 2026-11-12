"""Extractive sentence selection — the "no real LLM available" fallback from
Problem 2: pick the 1-3 most relevant sentences (not whole chunks) out of
already-relevant, already-markdown-clean text, instead of returning raw
chunks.

A pure relevance-score ranking tends to let one fact type (e.g. a transfer
limit) crowd out the other facts a customer actually asked for (e.g. the fee
and the speed), since real chunks often have several fact-bearing sentences
that are all about equally "relevant" by keyword overlap. So selection is
category-based: the sentence that most directly answers the question (highest
keyword overlap), then a fee fact (contains €) if any exists, then a timing
fact if any exists — deduplicated, capped at `max_sentences` and `max_words`.

The word budget is enforced in this priority order, not reading order: if the
first-priority pick is long, a lower-priority pick is dropped before a
higher-priority one would be, never the reverse. Only after the budget-aware
selection is finalized are the surviving sentences reordered for readability
to match their original position in the text.
"""
import re

from app.rag.relevance import TIME_WORDS, fact_score, keywords


def _split_sentences(text: str) -> list[str]:
    """Splits on sentence-ending punctuation, then re-merges a bare numbered
    -list marker ("1.", "12.") with the step text that follows it — plain
    splitting treats the period in "1. Open the app." as a second sentence
    boundary, producing the near-useless fragments "1." and "Open the app."
    A fixed-width lookbehind can't safely tell a list marker apart from a
    sentence ending right after a two-decimal amount (e.g. "costs €10.00."),
    since both look like "<digit>.<space><Capital>" — but only a genuine list
    marker ever surfaces as a BARE "<digit>." fragment with nothing else in
    it, so merging those after the fact is unambiguous and leaves real
    decimal-ending sentences untouched."""
    raw = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    merged: list[str] = []
    i = 0
    while i < len(raw):
        if re.fullmatch(r"\d{1,2}\.", raw[i]) and i + 1 < len(raw):
            merged.append(f"{raw[i]} {raw[i + 1]}")
            i += 2
        else:
            merged.append(raw[i])
            i += 1
    return merged


def extract_steps(text: str, max_steps: int = 4) -> list[str]:
    """Pulls numbered "How to" steps directly from policy text, in order,
    instead of generic sentence scoring — used when a procedural ("how do
    I...") question matches a document that actually has a steps list."""
    steps = [s for s in _split_sentences(text) if re.match(r"^\d+\.\s+\S", s)]
    return steps[:max_steps]


def _has_time_word(sentence: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", sentence.lower())) & TIME_WORDS)


def extract_best_sentences(
    query: str, text: str, max_sentences: int = 3, max_words: int | None = None
) -> list[str]:
    # Numbered "How to" steps are procedural fragments, not facts — they
    # belong only in a dedicated extract_steps() answer, never mixed into a
    # fact-based prose answer where "1. Open the app." would look like noise.
    sentences = [s for s in _split_sentences(text) if not re.match(r"^\d+\.\s", s)]
    if not sentences:
        return []

    q_kw = keywords(query)
    scored = [(s, len(q_kw & keywords(s)), fact_score(s)) for s in sentences]

    picked: list[str] = []
    word_count = 0

    def try_pick(predicate, key) -> bool:
        nonlocal word_count
        if len(picked) >= max_sentences:
            return False
        pool = [c for c in scored if c[0] not in picked and predicate(c)]
        if not pool:
            return False
        candidate = max(pool, key=key)[0]
        candidate_words = len(candidate.split())
        if max_words is not None and picked and word_count + candidate_words > max_words:
            return False  # a lower-priority pick must not bump a higher-priority one
        picked.append(candidate)
        word_count += candidate_words
        return True

    def try_pick_prefer_overlap(predicate, key) -> None:
        # A fee/time fact elsewhere in an otherwise-relevant document can be
        # about something unrelated to the actual question (e.g. a sentence
        # about *returned* transfers scoring high on "fee" + "business days"
        # for a query about *sending* money). Prefer a candidate that also
        # overlaps the query; only fall back to "any" if none does.
        if not try_pick(lambda c: predicate(c) and c[1] > 0, key):
            try_pick(predicate, key)

    # Fee facts first: the sentence with the highest raw keyword overlap is
    # often just whichever sentence happens to repeat the query's word, not
    # necessarily the most informative one — a concrete euro figure usually is.
    try_pick_prefer_overlap(lambda c: "€" in c[0], key=lambda c: c[2])  # a fee/amount fact
    try_pick(lambda c: c[1] > 0, key=lambda c: (c[1], c[2]))  # most directly answers the question
    try_pick_prefer_overlap(lambda c: _has_time_word(c[0]), key=lambda c: c[2])  # a timing fact

    if len(picked) < max_sentences:
        remaining = sorted(
            (c for c in scored if c[0] not in picked), key=lambda c: (c[1], c[2]), reverse=True
        )
        for s, overlap, fscore in remaining:
            if overlap + fscore == 0:
                break
            s_words = len(s.split())
            if max_words is not None and picked and word_count + s_words > max_words:
                continue
            picked.append(s)
            word_count += s_words
            if len(picked) >= max_sentences:
                break

    order = {s: i for i, s in enumerate(sentences)}
    picked.sort(key=lambda s: order[s])
    return picked
