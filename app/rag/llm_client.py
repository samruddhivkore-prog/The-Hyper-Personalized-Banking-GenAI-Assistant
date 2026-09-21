"""Pluggable LLM client. Provider selected by settings.llm_provider:

- stub    : deterministic, extractive, zero-network answer built only from the
            retrieved context. Used by default, in tests, and in CI so the
            grounding contract is verifiable without an API key.
- groq    : Groq free tier (Llama 3.1) — used for the deployed demo.
- gemini  : Google Gemini free tier — alternative deployed-demo provider.
- ollama  : local model — used during development to avoid burning API quota.

Every backend implements the same generate() contract so the RAG pipeline
(app/rag/pipeline.py) is provider-agnostic.
"""
import re
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings

_STOPWORDS = {
    "the", "a", "an", "is", "are", "to", "of", "for", "and", "or", "what",
    "how", "do", "i", "my", "in", "on", "with", "can", "does", "it", "this",
    "that", "you", "your", "be", "was", "were", "fastest", "way",
}


def _keywords(text: str) -> set[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 2}


class LLMClient(ABC):
    @abstractmethod
    def generate(self, prompt: str, context_chunks: list[str], temperature: float = 0.2) -> str:
        ...


class StubLLMClient(LLMClient):
    """Extractive, no-network 'LLM'. Never emits text absent from context_chunks
    (beyond fixed connective phrasing), which makes the no-hallucination
    contract mechanically testable."""

    def generate(self, prompt: str, context_chunks: list[str], temperature: float = 0.2) -> str:
        if not context_chunks:
            return "I don't have enough information to answer that."

        query_match = re.search(r"Question:\s*(.*)", prompt)
        query = query_match.group(1).strip() if query_match else ""
        q_keywords = _keywords(query)

        sentences: list[tuple[str, int]] = []
        for chunk in context_chunks:
            for sent in re.split(r"(?<=[.!?])\s+", chunk.strip()):
                sent = sent.strip()
                if not sent:
                    continue
                overlap = len(_keywords(sent) & q_keywords)
                sentences.append((sent, overlap))

        sentences.sort(key=lambda s: s[1], reverse=True)
        top = [s for s, score in sentences if score > 0][:3]
        if not top:
            top = [s for s, _ in sentences[:2]]

        return " ".join(top)


class GroqLLMClient(LLMClient):
    def __init__(self, api_key: str, model: str):
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self._model = model

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
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


def get_llm_client() -> LLMClient:
    settings = get_settings()
    provider = settings.llm_provider.lower()
    if provider == "groq" and settings.groq_api_key:
        return GroqLLMClient(settings.groq_api_key, settings.groq_model)
    if provider == "gemini" and settings.gemini_api_key:
        return GeminiLLMClient(settings.gemini_api_key, settings.gemini_model)
    if provider == "ollama":
        return OllamaLLMClient(settings.ollama_host, settings.ollama_model)
    return StubLLMClient()
