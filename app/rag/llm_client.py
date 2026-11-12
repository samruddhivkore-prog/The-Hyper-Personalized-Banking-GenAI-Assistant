"""Pluggable LLM client. Provider is chosen by the LLM_PROVIDER env var
(groq | gemini | anthropic | ollama | stub). Each provider's API key comes
from its own env var (GROQ_API_KEY / GEMINI_API_KEY / ANTHROPIC_API_KEY); if
the chosen provider has no key set, get_llm_client() falls back to the
deterministic, extractive, zero-network stub rather than erroring — this is
also what tests/CI use, so the grounding contract is verifiable without any
API key. LLM_MODEL overrides a provider's default model name if set.

Every backend implements the same generate() contract so the RAG pipeline
(app/rag/pipeline.py) is provider-agnostic, and each retries once on a
transient error before raising — app/rag/pipeline.py turns that into a
friendly message rather than a crash.
"""
import re
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.rag.extraction import extract_best_sentences
from app.rag.prompts import NOT_FOUND_MESSAGE


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, context_chunks: list[str], temperature: float = 0.2) -> str:
        ...


class StubLLMClient(LLMClient):
    """Extractive, no-network 'LLM'. Never emits text absent from
    context_chunks, which makes the no-hallucination contract mechanically
    testable. Used directly here only when called standalone (e.g. tests);
    app/rag/pipeline.py calls app.rag.extraction directly instead, so it can
    track which source document each picked sentence came from."""

    def generate(self, prompt: str, context_chunks: list[str], temperature: float = 0.2) -> str:
        if not context_chunks:
            return NOT_FOUND_MESSAGE

        query_match = re.search(r"Question:\s*(.*)", prompt)
        query = query_match.group(1).strip() if query_match else ""

        picked: list[str] = []
        for chunk in context_chunks:
            if len(picked) >= 3:
                break
            picked.extend(extract_best_sentences(query, chunk, max_sentences=3 - len(picked)))

        return " ".join(picked) if picked else NOT_FOUND_MESSAGE


class AnthropicLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        from anthropic import Anthropic

        self._client = Anthropic(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate(self, prompt: str, context_chunks: list[str], temperature: float = 0.0) -> str:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=300,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class GroqLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate(self, prompt: str, context_chunks: list[str], temperature: float = 0.2) -> str:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return completion.choices[0].message.content


class GeminiLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=8))
    def generate(self, prompt: str, context_chunks: list[str], temperature: float = 0.2) -> str:
        response = self._model.generate_content(
            prompt, generation_config={"temperature": temperature}
        )
        return response.text


class OllamaLLMClient(LLMClient):
    def __init__(self, host: str, model: str):
        self._host = host
        self._model = model

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
    def generate(self, prompt: str, context_chunks: list[str], temperature: float = 0.2) -> str:
        import httpx

        resp = httpx.post(
            f"{self._host}/api/generate",
            json={"model": self._model, "prompt": prompt, "stream": False,
                  "options": {"temperature": temperature}},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["response"]


def describe_llm_provider() -> str:
    """Human-readable line for the startup log and the admin app — never
    raises, since it must be safe to call before any provider is constructed."""
    settings = get_settings()
    provider = settings.llm_provider.lower()
    has_key = {
        "groq": bool(settings.groq_api_key),
        "gemini": bool(settings.gemini_api_key),
        "anthropic": bool(settings.anthropic_api_key),
    }
    if provider in has_key and has_key[provider]:
        return provider
    return "stub fallback (no API key)"


def get_llm_client() -> LLMClient:
    settings = get_settings()
    provider = settings.llm_provider.lower()
    if provider == "groq" and settings.groq_api_key:
        return GroqLLMClient(settings.groq_api_key, settings.llm_model or settings.groq_model)
    if provider == "gemini" and settings.gemini_api_key:
        return GeminiLLMClient(settings.gemini_api_key, settings.llm_model or settings.gemini_model)
    if provider == "anthropic" and settings.anthropic_api_key:
        return AnthropicLLMClient(settings.anthropic_api_key, settings.llm_model or settings.anthropic_model)
    if provider == "ollama":
        return OllamaLLMClient(settings.ollama_host, settings.llm_model or settings.ollama_model)
    return StubLLMClient()
