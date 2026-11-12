"""which_product handling: given a customer's stated purpose, pick the one
catalog product whose purposes match and answer using only that product's
own numbers — never a competing product's. The LLM picks and writes the
answer in one call; the picked name is validated against the real catalog
before anything is trusted, and the fallback/no-match path is deterministic."""
import json
import re
from functools import lru_cache
from pathlib import Path

from app.rag.llm_client import LLMClient, StubLLMClient
from app.rag.markdown_clean import strip_markdown
from app.rag.pipeline import OUT_OF_SCOPE_TERMS
from app.rag.prompts import NOT_FOUND_MESSAGE
from app.rag.relevance import keywords

CATALOG_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "product_catalog.json"

MATCH_PROMPT = (
    "Here is a bank's product catalog as JSON:\n{catalog}\n\n"
    "A customer asked: \"{query}\"\n\n"
    "Pick the ONE product from the catalog whose purposes best match what the customer wants. "
    "If none of the products fit, say so. Answer in exactly 2 sentences: which product it is and "
    "why it fits, then its key numbers from the catalog. Do not mention any other product from the "
    "catalog. Do not restate eligibility rules unless the customer asked about eligibility. "
    'Reply with ONLY a JSON object: {{"product": "<exact product name from the catalog, or null if '
    'none fit>", "answer": "<your 2-sentence answer, or exactly "{not_found}" if none fit>"}}.'
).replace("{not_found}", NOT_FOUND_MESSAGE)


@lru_cache
def load_catalog() -> list[dict]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _doc_title(doc_id: str) -> str:
    from scripts.index_kb import load_kb_docs

    for doc in load_kb_docs():
        if doc["doc_id"] == doc_id:
            return doc["source_title"]
    return doc_id


def _match_rule_based(query: str, catalog: list[dict]) -> dict:
    q_kw = keywords(query)
    best = None
    best_overlap = 0
    for product in catalog:
        purpose_kw = keywords(" ".join(product["purposes"]))
        overlap = len(q_kw & purpose_kw)
        if overlap > best_overlap:
            best, best_overlap = product, overlap
    if not best:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}
    answer = f"{best['name']} fits this. {best['key_numbers']}."
    return {"answer": answer, "sources": [_doc_title(best["policy_doc"])]}


def match_product(query: str, llm_client: LLMClient) -> dict:
    if OUT_OF_SCOPE_TERMS.search(query):
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    catalog = load_catalog()

    if isinstance(llm_client, StubLLMClient):
        return _match_rule_based(query, catalog)

    prompt = MATCH_PROMPT.format(catalog=json.dumps(catalog), query=query)
    try:
        raw = llm_client.generate(prompt, context_chunks=[], temperature=0.0)
    except Exception:
        return {
            "answer": "Sorry, I'm having trouble answering right now. Please try again shortly.",
            "sources": [],
        }

    data = _extract_json(raw)
    product_name = data.get("product")
    answer = strip_markdown(str(data.get("answer", "")).strip())

    valid_names = {p["name"] for p in catalog}
    if not product_name or product_name not in valid_names or NOT_FOUND_MESSAGE in answer:
        return {"answer": NOT_FOUND_MESSAGE, "sources": []}

    matched = next(p for p in catalog if p["name"] == product_name)
    return {"answer": answer, "sources": [_doc_title(matched["policy_doc"])]}
