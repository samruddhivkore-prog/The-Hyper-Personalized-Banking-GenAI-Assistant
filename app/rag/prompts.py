"""Prompt construction. The strict grounding instruction here is the single
most important correctness rule in the system: the model must answer only
from CONTEXT and the recommendation reason codes, and say so when it can't."""

SYSTEM_INSTRUCTION = (
    "You are a banking assistant. Answer ONLY using the CONTEXT below.\n"
    "If the CONTEXT does not contain the answer, say you don't have enough information.\n"
    "Do not mention products or facts that are not in the CONTEXT."
)


def format_recommendations(recommendations: list[dict]) -> str:
    if not recommendations:
        return "(none)"
    lines = [
        f"- {r['product']} (score={r['score']}, reason={r['reason_code']})"
        for r in recommendations
    ]
    return "\n".join(lines)


def format_context(chunks: list[str]) -> str:
    if not chunks:
        return "(no relevant context retrieved)"
    return "\n\n---\n\n".join(chunks)


def build_prompt(
    segment: str | None,
    recommendations: list[dict],
    retrieved_chunks: list[str],
    query: str,
) -> str:
    return (
        f"System: {SYSTEM_INSTRUCTION}\n\n"
        f"Customer segment: {segment or 'unknown'}\n"
        f"Relevant recommendations for this customer:\n{format_recommendations(recommendations)}\n\n"
        f"CONTEXT:\n{format_context(retrieved_chunks)}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )
