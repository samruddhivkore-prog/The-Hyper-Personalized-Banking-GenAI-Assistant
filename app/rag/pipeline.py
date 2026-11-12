"""RAG answer assembly: retrieve -> gate on relevance -> generate -> confidence.

Retrieval pulls a wide candidate pool (this KB is tiny — a few dozen chunks —
so "wide" is cheap and means every chunk of a relevant document is actually
considered, not just whichever single window the embedding happened to rank
highest). Candidates are filtered per-chunk by keyword overlap with the
query — not as one combined blob — then grouped by source document, and only
the top 3 distinct documents (by best chunk score) are used as context.

For the stub LLM (no real provider configured), sentence selection and
source attribution happen together here rather than inside the stub client,
so "sources" reflects exactly which document each cited sentence came from —
this is what keeps an ambiguous query like "how much loan can I get" from
citing a same-keyword-but-wrong-product document (see tests/test_api.py).
For a real LLM provider, the top 3 documents' text is sent as context per the
exact prompt contract in app/rag/prompts.py, and "sources" is those 3
documents (standard RAG citation — a real LLM's exact sentence provenance
isn't mechanically trackable the way the stub's is).
"""
import re
from dataclasses import dataclass, field

from app.rag.embeddings import EmbeddingModel, get_embedding_model
from app.rag.extraction import extract_best_sentences, extract_steps
from app.rag.llm_client import LLMClient, StubLLMClient, get_llm_client
from app.rag.markdown_clean import strip_markdown
from app.rag.prompts import NOT_FOUND_MESSAGE, build_prompt
from app.rag.relevance import keywords
from app.rag.vector_store import RetrievedChunk, get_vector_store
from app.services.intent import is_how_to

MAX_STEPS = 4

CANDIDATE_POOL_SIZE = 20  # ~the whole KB in this demo; filtering does the real work
MAX_SOURCES = 3
MAX_SENTENCES = 3
MAX_ANSWER_WORDS = 55
MIN_USEFUL_WORD_BUDGET = 8  # below this, no further source can add a meaningful sentence
TRAILING_SOURCE_MIN_SCORE_RATIO = 0.65  # a 2nd/3rd source must score reasonably close to the top one

# A single shared keyword, by itself, isn't strong evidence of relevance (a
# credit card doc mentioning "customers who travel internationally" once
# shouldn't match "do you offer travel insurance" just because both say
# "travel"). Embedding score serves as a secondary check for weak
# single-keyword matches. The hash-fallback embedding's cosine scores run
# much lower than real sentence-transformers scores even for genuine
# matches, so the floor is backend-dependent — calibrated empirically
# against this KB (see tests/test_grounding_quality.py).
MIN_SCORE_BY_BACKEND = {"hash-fallback": 0.08, "sentence-transformers": 0.18}
DEFAULT_MIN_SCORE = 0.15

# Categories the bank deliberately never offers (see PART 1 of the product
# spec: "do NOT add travel insurance, car insurance, pet insurance or crypto
# trading — these stay as not covered"). A query naming one of these can
# still share an incidental word with an unrelated product (e.g. "car
# insurance" shares "car" with the Personal Loan doc's "buying a car"),
# which keyword+embedding retrieval alone can't reliably tell apart from a
# genuine match — so this is an explicit, permanent veto, not a heuristic.
OUT_OF_SCOPE_TERMS = re.compile(r"\binsurance\b|\bcrypto\w*\b|\bbitcoin\b", re.IGNORECASE)


@dataclass
class AssistantAnswer:
    answer: str
    sources: list[str]
    confidence: str  # "high" | "medium" | "none"
    retrieved_chunks: list[str] = field(default_factory=list)  # admin-only debugging info


def _confidence(top_score: float) -> str:
    return "high" if top_score >= 0.5 else "medium"


def _group_by_source(
    query: str, retrieved: list[RetrievedChunk], min_score: float
) -> list[tuple[str, str, float]]:
    """Per-chunk relevance gate (keyword overlap AND a minimum embedding
    score), then group surviving chunks by source document. Returns up to
    MAX_SOURCES (source_title, merged_text, best_score) tuples, ranked by
    best_score descending.

    If two or more sources match the query on exactly the same word(s), and
    that word is also literally part of each of those sources' own titles,
    it's too generic to tell them apart (e.g. "loan" matches both "Personal
    Loan" and "Student Loan Refinance" for a query that names neither) — only
    the highest-scoring one is kept. A source that merely mentions the other
    product by name in passing, without the word being core to its own
    title, is unaffected."""
    q_kw = keywords(query)
    texts_by_source: dict[str, list[str]] = {}
    best_score_by_source: dict[str, float] = {}
    overlap_by_source: dict[str, set[str]] = {}

    for chunk in retrieved:
        overlap = q_kw & keywords(chunk.text)
        if not overlap:
            continue
        if chunk.score < min_score:
            continue
        texts_by_source.setdefault(chunk.source_title, [])
        if chunk.text not in texts_by_source[chunk.source_title]:
            texts_by_source[chunk.source_title].append(chunk.text)
        best_score_by_source[chunk.source_title] = max(
            best_score_by_source.get(chunk.source_title, 0.0), chunk.score
        )
        overlap_by_source.setdefault(chunk.source_title, set()).update(overlap)

    same_generic_overlap: dict[frozenset, list[str]] = {}
    for source, overlap in overlap_by_source.items():
        if overlap <= keywords(source):
            same_generic_overlap.setdefault(frozenset(overlap), []).append(source)

    suppressed: set[str] = set()
    for competing in same_generic_overlap.values():
        if len(competing) > 1:
            ranked = sorted(competing, key=lambda s: best_score_by_source[s], reverse=True)
            suppressed.update(ranked[1:])

    ranked_sources = sorted(
        (s for s in best_score_by_source if s not in suppressed),
        key=best_score_by_source.get,
        reverse=True,
    )
    return [
        (source, " ".join(texts_by_source[source]), best_score_by_source[source])
        for source in ranked_sources[:MAX_SOURCES]
    ]


def answer_query(
    query: str,
    k: int = CANDIDATE_POOL_SIZE,
    embedding_model: EmbeddingModel | None = None,
    llm_client: LLMClient | None = None,
    vector_store=None,
) -> AssistantAnswer:
    if OUT_OF_SCOPE_TERMS.search(query):
        return AssistantAnswer(answer=NOT_FOUND_MESSAGE, sources=[], confidence="none")

    embedding_model = embedding_model or get_embedding_model()
    llm_client = llm_client or get_llm_client()
    vector_store = vector_store or get_vector_store()

    query_vec = embedding_model.encode_one(query)
    retrieved = vector_store.similarity_search(query_vec, k=k)

    min_score = MIN_SCORE_BY_BACKEND.get(embedding_model.backend, DEFAULT_MIN_SCORE)
    groups = _group_by_source(query, retrieved, min_score)
    if not groups:
        return AssistantAnswer(answer=NOT_FOUND_MESSAGE, sources=[], confidence="none")

    if is_how_to(query):
        # Deterministic step extraction, regardless of provider: a real LLM
        # asked to "write numbered steps" won't reliably stick to the exact
        # "1. "/"2. " format or stop at 4, but lifting the steps straight out
        # of the policy text does, by construction — and it's one method
        # only (whichever document ranked top), never a mix of two.
        top_source, top_text, top_score = groups[0]
        steps = extract_steps(top_text, max_steps=MAX_STEPS)
        if steps:
            # Steps come from already markdown-clean indexed text (see
            # app/rag/chunking.py) — join with newlines, not strip_markdown's
            # single-space join, so each numbered step stays on its own line.
            return AssistantAnswer(
                answer="\n".join(steps),
                sources=[top_source],
                confidence=_confidence(top_score),
                retrieved_chunks=[top_text],
            )
        # no steps in the matched doc — fall through to a normal 2-3 sentence answer

    if isinstance(llm_client, StubLLMClient):
        picked: list[tuple[str, str]] = []  # (sentence, source_title)
        word_count = 0
        top_score = groups[0][2]
        for source, text, score in groups:
            if len(picked) >= MAX_SENTENCES:
                break
            if picked and score < top_score * TRAILING_SOURCE_MIN_SCORE_RATIO:
                # A much weaker-scoring source shouldn't be used just to pad
                # the sentence count out to MAX_SENTENCES — a short, correct
                # answer beats padding it with a document that only matched
                # the query on an incidental, generic mention.
                break
            word_budget = MAX_ANSWER_WORDS - word_count
            if picked and word_budget < MIN_USEFUL_WORD_BUDGET:
                # Too little budget left to add anything meaningful from a
                # further source — stop rather than force in an oversized
                # sentence the next call's own "always keep a first pick"
                # rule would otherwise let through.
                break
            sentence_budget = MAX_SENTENCES - len(picked)
            for sentence in extract_best_sentences(
                query, text, max_sentences=sentence_budget, max_words=word_budget
            ):
                picked.append((sentence, source))
                word_count += len(sentence.split())

        if not picked:
            return AssistantAnswer(answer=NOT_FOUND_MESSAGE, sources=[], confidence="none")

        answer_text = strip_markdown(" ".join(s for s, _ in picked))
        sources = list(dict.fromkeys(src for _, src in picked))
        used_texts = [text for source, text, _ in groups if source in sources]
    else:
        context_texts = [text for _, text, _ in groups]
        prompt = build_prompt(context_texts, query)
        try:
            raw_answer = llm_client.generate(prompt, context_chunks=context_texts, temperature=0.0)
        except Exception:
            # Already retried once inside the client (see app/rag/llm_client.py);
            # a customer gets a friendly message, never a crash or a stack trace.
            return AssistantAnswer(
                answer="Sorry, I'm having trouble answering right now. Please try again shortly.",
                sources=[],
                confidence="none",
            )
        answer_text = strip_markdown(raw_answer.strip())
        if NOT_FOUND_MESSAGE in answer_text:
            return AssistantAnswer(answer=NOT_FOUND_MESSAGE, sources=[], confidence="none")

        # Only cite a document the answer actually draws on — a document
        # merely sent as context isn't necessarily what the model used,
        # especially with several candidate documents offered at once (e.g.
        # Personal Loan, Student Loan, and Home Loan all share the word
        # "loan"). Require a keyword the answer shares with THIS candidate
        # but not with any of the other candidates — a word every candidate
        # has in common proves nothing about which one was actually cited.
        answer_kw = keywords(answer_text)
        kw_by_source = {source: keywords(text) for source, text, _ in groups}
        used = []
        for source, text, _ in groups:
            other_kw = set().union(*(kw for s, kw in kw_by_source.items() if s != source), set())
            distinctive = kw_by_source[source] - other_kw
            if answer_kw & distinctive:
                used.append((source, text))
        if not used:
            used = [(groups[0][0], groups[0][1])]
        sources = [source for source, _ in used][:MAX_SOURCES]
        used_texts = [text for _, text in used][:MAX_SOURCES]

    return AssistantAnswer(
        answer=answer_text,
        sources=sources,
        confidence=_confidence(groups[0][2]),
        retrieved_chunks=used_texts,
    )
