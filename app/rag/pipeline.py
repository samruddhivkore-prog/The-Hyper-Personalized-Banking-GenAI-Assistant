"""RAG answer assembly: retrieve -> build prompt -> generate -> confidence.

This is the function that implements step 6 of HandleRecommendationRequest
(spec section 8.2): if nothing relevant is retrieved, the system says so
instead of letting the LLM guess.
"""
from dataclasses import dataclass

from app.rag.embeddings import EmbeddingModel, get_embedding_model
from app.rag.llm_client import LLMClient, get_llm_client
from app.rag.prompts import build_prompt
from app.rag.vector_store import RetrievedChunk, get_vector_store


@dataclass
class AssistantAnswer:
    answer: str
    sources: list[str]
    confidence: str  # "high" | "medium" | "low" | "none"


def _confidence(retrieved: list[RetrievedChunk]) -> str:
    if not retrieved:
        return "none"
    top_score = retrieved[0].score
    if top_score >= 0.55:
        return "high"
    if top_score >= 0.35:
        return "medium"
    return "low"


def answer_query(
    query: str,
    segment: str | None,
    recommendations: list[dict],
    k: int = 4,
    embedding_model: EmbeddingModel | None = None,
    llm_client: LLMClient | None = None,
    vector_store=None,
) -> AssistantAnswer:
    embedding_model = embedding_model or get_embedding_model()
    llm_client = llm_client or get_llm_client()
    vector_store = vector_store or get_vector_store()

    query_vec = embedding_model.encode_one(query)
    retrieved = vector_store.similarity_search(query_vec, k=k)

    if not retrieved:
        return AssistantAnswer(
            answer="I don't have enough information to answer that.",
            sources=[],
            confidence="none",
        )

    prompt = build_prompt(segment, recommendations, [c.text for c in retrieved], query)
    answer_text = llm_client.generate(prompt, context_chunks=[c.text for c in retrieved])

    sources = list(dict.fromkeys(c.source_title for c in retrieved))  # de-dup, keep order
    return AssistantAnswer(
        answer=answer_text,
        sources=sources,
        confidence=_confidence(retrieved),
    )
