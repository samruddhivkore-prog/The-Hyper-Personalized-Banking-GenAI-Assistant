import re

from app.rag.llm_client import StubLLMClient
from app.rag.prompts import build_prompt


def test_build_prompt_includes_all_sections():
    prompt = build_prompt(
        segment="digital_first_young_professional",
        recommendations=[
            {"product": "Real-Time Payments", "score": 0.92, "reason_code": "high_transfer_frequency"}
        ],
        retrieved_chunks=["Real-Time Payments costs $1.00 per transfer."],
        query="What's the fastest way to send $2,000?",
    )
    assert "digital_first_young_professional" in prompt
    assert "Real-Time Payments" in prompt
    assert "high_transfer_frequency" in prompt
    assert "$1.00 per transfer" in prompt
    assert "What's the fastest way to send $2,000?" in prompt
    assert "Answer ONLY using the CONTEXT" in prompt


def test_build_prompt_handles_no_recommendations_and_no_context():
    prompt = build_prompt(segment=None, recommendations=[], retrieved_chunks=[], query="hello")
    assert "(none)" in prompt
    assert "(no relevant context retrieved)" in prompt
    assert "unknown" in prompt


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def test_stub_llm_never_invents_words_outside_context():
    context = [
        "Real-Time Payments costs a flat fee of $1.00 per transfer and arrives within seconds.",
        "Wire transfers cost $25.00 and arrive same-day during business hours.",
    ]
    prompt = build_prompt(
        segment="digital_first",
        recommendations=[],
        retrieved_chunks=context,
        query="What's the fastest way to send $2,000 to my sister's account?",
    )
    llm = StubLLMClient()
    answer = llm.generate(prompt, context_chunks=context)

    context_words = set()
    for chunk in context:
        context_words |= _words(chunk)
    answer_words = _words(answer)

    assert answer_words.issubset(context_words)


def test_stub_llm_says_it_does_not_know_with_empty_context():
    llm = StubLLMClient()
    answer = llm.generate("irrelevant prompt", context_chunks=[])
    assert "don't have enough information" in answer.lower()
