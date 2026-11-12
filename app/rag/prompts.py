"""Prompt construction for a real LLM provider (Anthropic/Groq/Gemini/Ollama).
The stub provider (app/rag/extraction.py) does not use this — it's
extractive, not generative — but shares the same NOT_FOUND_MESSAGE so both
paths degrade identically from a customer's point of view."""

NOT_FOUND_MESSAGE = "I couldn't find this in our policies. Please contact the bank directly."

INSTRUCTION = (
    "Answer the customer's question in 2-3 short plain sentences, using ONLY the text below. "
    "Answer the exact question first. Give numbers (fee, limit, time) only if they appear in "
    "the text. If the text does not directly answer the question, reply exactly: "
    + NOT_FOUND_MESSAGE
    + " A related but different product does not count as an answer (a personal loan is not a "
    "mortgage). If several methods exist and the customer asks for the fastest or cheapest, "
    "pick that one. No markdown, no headings, no bullet points."
)


def build_prompt(context_chunks: list[str], query: str) -> str:
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant text)"
    return f"{INSTRUCTION}\n\nTEXT:\n{context}\n\nQuestion: {query}\nAnswer:"
