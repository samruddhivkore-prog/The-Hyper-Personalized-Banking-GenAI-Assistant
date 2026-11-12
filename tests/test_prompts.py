from app.rag.extraction import extract_best_sentences
from app.rag.llm_client import StubLLMClient
from app.rag.prompts import NOT_FOUND_MESSAGE, build_prompt


def test_build_prompt_includes_instruction_context_and_question():
    prompt = build_prompt(
        context_chunks=["Real-Time Payments costs €1.00 per transfer."],
        query="What's the fastest way to send €2,000?",
    )
    assert "2-3 short plain sentences" in prompt
    assert "using ONLY the text below" in prompt
    assert NOT_FOUND_MESSAGE in prompt
    assert "€1.00 per transfer" in prompt
    assert "What's the fastest way to send €2,000?" in prompt


def test_build_prompt_handles_no_context():
    prompt = build_prompt(context_chunks=[], query="hello")
    assert "(no relevant text)" in prompt


def test_extract_best_sentences_only_uses_words_from_the_text():
    import re

    text = (
        "Real-Time Payments costs a flat fee of €1.00 per transfer and arrives "
        "within seconds. Wire transfers cost €25.00 and arrive same-day."
    )
    picked = extract_best_sentences("What's the fastest way to send money?", text, max_sentences=3)
    assert picked

    text_words = set(re.findall(r"[a-z0-9]+", text.lower()))
    for sentence in picked:
        sentence_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        assert sentence_words.issubset(text_words)


def test_extract_best_sentences_empty_text_returns_nothing():
    assert extract_best_sentences("anything", "", max_sentences=3) == []


def test_stub_llm_says_it_does_not_know_with_empty_context():
    llm = StubLLMClient()
    answer = llm.generate("Question: anything\nAnswer:", context_chunks=[])
    assert answer == NOT_FOUND_MESSAGE
