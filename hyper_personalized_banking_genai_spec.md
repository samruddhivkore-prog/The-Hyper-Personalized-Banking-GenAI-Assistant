# Hyper-Personalized Banking GenAI Assistant — Project Specification

**Version:** 1.0
**Type:** End-to-end, production-ready AI/ML portfolio project
**Domain:** Financial services — CRM hyper-personalization powered by GenAI

---

## 1. Overview

A backend service + assistant that takes a bank's customer data, segments customers,
generates personalized product/action recommendations, and answers customer questions
in natural language — with every answer grounded in real bank policy documents (via
RAG) rather than invented by the model. The system also audits its own recommendations
for fairness across customer segments before they go out.

In one sentence: **given a customer and (optionally) a question, return personalized,
explainable, policy-grounded recommendations through a real API, with tests,
containers, CI/CD, and a live deployed demo.**

---

## 2. Motivation

Financial institutions increasingly rely on AI/ML to move CRM from static segmentation
to real-time, individualized interaction — personalized content, product
recommendations, and channel choices tailored per customer. GenAI adds a second layer
on top of this: it can explain *why* a recommendation was made, answer free-form
questions against internal policy documents, and reduce the manual work of navigating
long product manuals (for both customers and staff).

Two risks come with this power, and both are central to how banks evaluate AI systems:

1. **Hallucination / ungrounded answers** — a wrong answer about a real financial
   product is a compliance and trust problem, not just a UX bug. This is why the
   assistant layer must be RAG-grounded, not a free-form LLM call.
2. **Unfair treatment across customer segments** — a recommendation engine that
   systematically favors one group is a regulatory risk. This is why a fairness
   audit is a first-class layer, not an afterthought.

This project exists to demonstrate — with a working, deployed system — that these two
concerns can be engineered in from the start, not bolted on. That is the story the
project tells on a resume: not "I called an LLM API," but "I built a personalization
system for a regulated domain and treated grounding and fairness as engineering
requirements."

---

## 3. Approach

The system is built in four layers, mirroring how hyper-personalization strategies are
structured in practice:

| Layer | Question it answers | Component |
|---|---|---|
| Data Foundation | Who is this customer? | Customer profile store (Postgres) |
| Decisions | What should we recommend, and is it fair? | Segmentation + recommender + fairness auditor |
| Design | How do we explain it without making things up? | RAG-grounded GenAI response layer |
| Distribution | How does the customer actually receive it? | FastAPI + thin frontend |

Design philosophy:

- **Deterministic where possible, generative only where needed.** Segmentation and
  scoring are classical ML (clustering, scoring functions) — cheap, fast, explainable,
  and auditable. The LLM is used only for the parts that genuinely need language:
  answering free-text questions and turning a recommendation list into a
  natural-language explanation.
- **Retrieval before generation.** The LLM never answers a policy question from its
  own knowledge. It answers from documents retrieved at request time. If nothing
  relevant is retrieved, the system says so instead of guessing.
- **Every layer is a service, not a notebook.** Each of the four layers above is a
  Python module with a defined input/output contract, independently testable.
- **Production concerns are part of the spec, not "phase 2."** Tests, containerization,
  CI/CD, and deployment are treated as required components of the project, same as the
  ML logic.

---

## 4. System Architecture

```
                                   ┌─────────────────────┐
                                   │   Frontend (chat UI) │
                                   │  Streamlit / React   │
                                   └──────────┬───────────┘
                                              │ HTTPS
                                   ┌──────────▼───────────┐
                                   │      FastAPI app       │
                                   │  /recommendations      │
                                   │  /chat                 │
                                   │  /segments              │
                                   │  /health                │
                                   └───┬─────────────┬──────┘
                     ┌─────────────────┘             └────────────────────┐
          ┌──────────▼──────────┐                          ┌──────────────▼─────────────┐
          │   Decisions layer    │                          │       Design layer          │
          │ ─────────────────── │                          │ ──────────────────────────  │
          │ SegmentationModel     │                          │ Embedding model              │
          │ RecommenderModel      │                          │ Vector store (Chroma)        │
          │ FairnessAuditor       │                          │ LLM client (Groq/Gemini free)│
          └──────────┬───────────┘                          └──────────────┬───────────────┘
                     │                                                     │
          ┌──────────▼───────────┐                          ┌──────────────▼───────────────┐
          │  Data Foundation      │                          │   Knowledge base source docs  │
          │  Postgres (customers, │                          │   (product policies, FAQs)     │
          │  transactions,        │                          │   indexed offline into Chroma  │
          │  products, scores)     │                          └───────────────────────────────┘
          └────────────────────────┘

          Cross-cutting: structured logging, /health, Sentry error tracking,
          GitHub Actions CI/CD, Docker/docker-compose, deployed on Render/Fly.io + HF Spaces.
```

---

## 5. Input Specification (Data Foundation)

### 5.1 Source datasets (free, public)

Use one or combine two of the following, remapped into the schema below:

- **Kaggle — "Bank Customer Segmentation"** (demographics + account behavior)
- **Kaggle — "Lending Club Loan Data"** (loan products, credit behavior)
- **IBM TabFormer (GitHub: IBM/TabFormer)** — synthetic transaction data, useful for
  realistic transaction volume/patterns without any real customer data

### 5.2 Customer profile schema (Postgres table: `customers`)

```json
{
  "customer_id": "CUST00123",
  "age": 29,
  "income_bracket": "50k_75k",
  "account_tenure_months": 34,
  "products_held": ["checking", "savings"],
  "channel_preference": "mobile",
  "avg_monthly_transactions": 42,
  "recent_transactions": [
    {"date": "2026-08-30", "category": "transfer", "amount": 250.00},
    {"date": "2026-08-28", "category": "bill_pay", "amount": 89.50}
  ]
}
```

### 5.3 API request schema

```json
POST /recommendations
{
  "customer_id": "CUST00123",
  "query": "What's the fastest way to send $2,000 to my sister's account?"
}
```

`query` is optional. If omitted, the endpoint returns recommendations and a default
personalized summary with no free-text answer.

---

## 6. Middle Layers (Decisions + Design)

### 6.1 Customer Segmentation

- **Input:** customer profile features (age, income bracket, tenure, transaction
  frequency, channel preference — numerically encoded).
- **Method:** KMeans (start with k=5, validate with silhouette score) or, if you want a
  stronger resume line, embed the profile as a vector and cluster in embedding space.
- **Output:** `segment_id` and a human-readable label, e.g.
  `"digital_first_young_professional"`.
- **Stored:** written back to `customers.segment_id`, refreshed on a schedule (batch
  job), not per-request.

### 6.2 Recommendation / Next-Best-Action Engine

- **Input:** customer profile + segment + product catalog.
- **Method:** start simple — a rules/scoring function (e.g., weighted match between
  customer attributes and product eligibility/relevance), upgrade to collaborative
  filtering if time allows.
- **Output:** ranked list of `{product, score, reason_code}`.
- **Constraint:** every recommendation must carry a `reason_code` (e.g.
  `"high_transfer_frequency"`) — this is what lets the GenAI layer explain the
  recommendation truthfully instead of inventing a justification.

### 6.3 Fairness Auditor

- **Input:** batch of `(segment, recommendation)` pairs over a recent window.
- **Method:** compute demographic parity difference — recommendation rate for a given
  product should not differ beyond a threshold (e.g. 10 percentage points) across
  segments defined by protected-adjacent attributes like age bracket or income
  bracket, unless justified by an eligibility rule.
- **Output:** `fairness_flags: []` (empty if within threshold) attached to the
  response, and a periodic report artifact (CSV/JSON) for the README.
- **This is the layer that ties back to dissertation-level fairness work** — frame it
  in the README as "fairness-by-design," not a compliance afterthought.

### 6.4 RAG Knowledge Base

- **Source documents:** product policy pages, FAQs, fee schedules (write 10–15
  realistic synthetic documents if no public corpus is used — e.g. "Real-Time Payments
  FAQ," "Wire Transfer Fees," "ACH Transfer Policy," "Personal Loan Eligibility").
- **Indexing (offline job):** chunk documents (~300–500 tokens, with overlap), embed
  with `sentence-transformers/all-MiniLM-L6-v2` (free, local), store in Chroma with
  `{doc_id, chunk_id, source_title}` metadata.
- **Retrieval (runtime):** embed the user query, similarity search top-k (k=4),
  return chunks + source titles.

### 6.5 GenAI Response Generation

- **Input:** customer profile, top-k recommendations with reason codes, retrieved
  document chunks, user query.
- **Prompt contract (strict):** the prompt must instruct the model to answer **only**
  from the retrieved chunks and the recommendation reason codes, and to say
  "I don't have enough information to answer that" if the retrieved context doesn't
  cover the question — this is the single most important correctness rule in the
  whole system.
- **LLM:** a free-tier hosted model (Groq's free tier running Llama 3.1, or Google
  Gemini free tier) for the deployed demo; Ollama locally during development to avoid
  burning API quota while testing.
- **Output:** `{answer, sources: [source_title, ...], confidence}`.

---

## 7. Output Specification

### 7.1 API response schema

```json
{
  "customer_id": "CUST00123",
  "segment": "digital_first_young_professional",
  "recommendations": [
    {
      "product": "Real-Time Payments",
      "score": 0.92,
      "reason_code": "high_transfer_frequency"
    },
    {
      "product": "High-Yield Savings",
      "score": 0.81,
      "reason_code": "stable_balance_growth"
    }
  ],
  "assistant_response": {
    "answer": "For sending $2,000 quickly, Real-Time Payments is your fastest option — funds typically arrive within seconds, for a $1 fee. A standard ACH transfer is free but takes 1-3 business days. Wire transfer arrives same-day for a $25 fee.",
    "sources": ["Real-Time Payments FAQ", "Wire Transfer Fees", "ACH Transfer Policy"],
    "confidence": "high"
  },
  "fairness_flags": [],
  "timestamp": "2026-09-10T14:22:03Z"
}
```

### 7.2 Ideal output — full walkthrough example

**Customer:** CUST00123, 29, digital-first segment, high transfer frequency,
2.8-year tenure, mobile-first.

**Request:**
```
POST /recommendations
{ "customer_id": "CUST00123", "query": "What's the fastest way to send $2,000 to my sister's account?" }
```

**What happens internally (trace):**
1. Profile fetched from Postgres.
2. Segment already computed: `digital_first_young_professional`.
3. Recommender returns Real-Time Payments (0.92) and High-Yield Savings (0.81), each
   with a reason code.
4. Fairness auditor checks this recommendation against the last 24h of recommendations
   for this segment vs. others — no flag raised.
5. Query embedded, top-4 chunks retrieved from Chroma: two from "Real-Time Payments
   FAQ," one from "Wire Transfer Fees," one from "ACH Transfer Policy."
6. Prompt assembled with profile + reason codes + retrieved chunks + query, sent to
   the LLM.
7. LLM returns an answer grounded only in the retrieved fee/timing facts.
8. Response assembled and returned; interaction logged.

**What "good" looks like:** the answer names real numbers that exist verbatim in the
source documents (fees, timing), attributes them to the correct product, and does not
mention any product that wasn't in the retrieved context. If you manually delete the
"Real-Time Payments FAQ" document and re-run the same query, the answer should
visibly degrade to "I don't have information on instant transfer options" rather than
guessing — that behavior is the actual test of whether the RAG grounding works.

---

## 8. End-to-End Algorithm

### 8.1 Offline / batch pipeline (runs on a schedule, not per-request)

```
ALGORITHM: OfflineRefresh()

1. profiles ← DB.fetch_all_customer_profiles()
2. features ← Featurize(profiles)                      # encode age, income, tenure, etc.
3. segment_model ← KMeans(k=5).fit(features)
4. FOR EACH profile IN profiles:
       profile.segment_id ← segment_model.predict(profile.features)
       DB.update_customer_segment(profile.customer_id, profile.segment_id)
5. FOR EACH profile IN profiles:
       candidates ← RecommenderModel.score(profile, all_products)
       DB.store_recommendations(profile.customer_id, candidates)
6. recent_recs ← DB.fetch_recent_recommendations(window="24h")
7. fairness_report ← FairnessAuditor.compute_parity(recent_recs, group_by="segment")
8. Log.write(fairness_report, path="reports/fairness_YYYYMMDD.json")
9. IF new_or_changed_docs EXISTS:
       chunks ← ChunkDocuments(product_kb_docs)
       embeddings ← EmbeddingModel.encode(chunks)
       VectorStore.upsert(chunks, embeddings, metadata)
```

### 8.2 Online / runtime pipeline (per API request)

```
ALGORITHM: HandleRecommendationRequest(customer_id, query)

1. profile ← DB.fetch_customer_profile(customer_id)
2. IF profile IS NULL:
       RETURN HTTP_404("customer not found")

3. segment ← profile.segment_id
4. top_k ← DB.fetch_stored_recommendations(customer_id, k=3)

5. fairness_flags ← FairnessAuditor.check_against_latest_report(segment, top_k)

6. IF query IS NOT NULL:
       q_vec ← EmbeddingModel.encode(query)
       retrieved ← VectorStore.similarity_search(q_vec, k=4)
       IF retrieved IS EMPTY:
           answer ← "I don't have enough information to answer that."
           sources ← []
       ELSE:
           prompt ← BuildPrompt(profile, top_k, retrieved, query)
           llm_out ← LLM.generate(prompt, temperature=0.2)
           answer ← llm_out.text
           sources ← [doc.source_title FOR doc IN retrieved]
   ELSE:
       answer ← TemplateEngine.render("default_summary", profile, top_k)
       sources ← []

7. response ← {
       customer_id: customer_id,
       segment: segment,
       recommendations: top_k,
       assistant_response: { answer: answer, sources: sources, confidence: Confidence(retrieved) },
       fairness_flags: fairness_flags,
       timestamp: Now()
   }

8. Logger.log_interaction(response)
9. RETURN HTTP_200(response)
```

### 8.3 Prompt template (used in step 6 above)

```
System: You are a banking assistant. Answer ONLY using the CONTEXT below.
If the CONTEXT does not contain the answer, say you don't have enough information.
Do not mention products or facts that are not in the CONTEXT.

Customer segment: {segment}
Relevant recommendations for this customer: {top_k_with_reason_codes}

CONTEXT:
{retrieved_chunks}

Question: {query}
Answer:
```

---

## 9. Build Roadmap — Steps to Achieve

1. **Architecture & schema first.** Write the diagram (Section 4) and the DB schema
   (Section 5.2) before any code. Create the GitHub repo, README skeleton.
2. **Data foundation.** Load a chosen dataset into Postgres matching the schema.
   Write 10–15 synthetic product policy documents for the knowledge base.
3. **Segmentation + recommender (offline pipeline, Section 8.1, steps 1–5).** Test on
   a sample of profiles; sanity-check segment labels make sense.
4. **Fairness auditor (offline pipeline, steps 6–8).** Confirm it produces a report
   and correctly flags an intentionally-biased test case (write one on purpose to
   verify the check works).
5. **RAG indexing (offline pipeline, step 9).** Chunk + embed + store the KB docs;
   manually query the vector store to confirm retrieval quality before wiring in
   the LLM.
6. **Online pipeline (Section 8.2).** Implement `HandleRecommendationRequest` as a
   plain Python function first, callable from a script — no API yet.
7. **Wrap it in FastAPI.** `/recommendations`, `/chat`, `/segments`, `/health`
   endpoints; Pydantic request/response models matching Sections 5.3 and 7.1.
8. **Tests.** Unit tests for segmentation, recommender, fairness auditor, prompt
   builder. Integration tests hitting the FastAPI endpoints (use a test DB).
9. **CI.** GitHub Actions workflow: install deps, run lint (ruff/flake8), run pytest,
   on every push and PR.
10. **Containerize.** Dockerfile for the API; docker-compose with API + Postgres +
    Chroma, so `docker-compose up` runs the whole stack locally.
11. **Frontend.** Minimal Streamlit chat UI calling the API (or React if you want the
    extra resume line).
12. **Deploy.** API + Postgres on Render/Fly.io free tier + Supabase/Neon; frontend on
    Hugging Face Spaces. Confirm the live link works end to end.
13. **Polish.** Structured logging, Sentry (free tier) for error tracking, README with
    architecture diagram + demo GIF + the "delete a doc and watch it degrade
    gracefully" test as a documented example of the grounding actually working.
14. **Write the case study.** One page: problem, architecture, one hard decision you
    made and why, one metric you measured, link to the live demo.

---

## 10. Tech Stack

| Concern | Choice | Why |
|---|---|---|
| API framework | FastAPI | async, auto docs, Pydantic validation |
| Database | Postgres (Supabase/Neon free tier) | real relational DB, free hosted tier |
| Vector store | Chroma | free, embeddable, simple to self-host |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | free, local, no API cost |
| LLM | Groq (Llama 3.1) or Gemini free tier | free tier, fast, good enough for grounded QA |
| Frontend | Streamlit (or React) | fast to build; React if extra polish wanted |
| Containers | Docker + docker-compose | portability, one-command local setup |
| CI/CD | GitHub Actions | free for public repos |
| Hosting (API) | Render or Fly.io free tier | free, supports Docker deploys |
| Hosting (frontend) | Hugging Face Spaces | free, good for ML demos |
| Error tracking | Sentry free tier | production-grade signal at no cost |
| Testing | pytest | standard, well supported |

---

## 11. Reference Code Skeletons

### 11.1 Dockerfile

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 11.2 docker-compose.yml

```yaml
version: "3.9"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    depends_on:
      - db
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app
      POSTGRES_DB: banking_assistant
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
volumes:
  pgdata:
```

### 11.3 GitHub Actions CI (`.github/workflows/ci.yml`)

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install -r requirements.txt
      - run: ruff check .
      - run: pytest -v
```

### 11.4 FastAPI endpoint skeleton (`app/main.py`)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Hyper-Personalized Banking Assistant")

class RecommendationRequest(BaseModel):
    customer_id: str
    query: str | None = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/recommendations")
def get_recommendations(req: RecommendationRequest):
    profile = fetch_customer_profile(req.customer_id)
    if profile is None:
        raise HTTPException(status_code=404, detail="customer not found")
    return handle_recommendation_request(profile, req.query)
```

---

## 12. Evaluation Metrics

| Layer | Metric | Target |
|---|---|---|
| Segmentation | Silhouette score | > 0.3 as a sanity floor |
| Recommender | Precision@3 on held-out interactions (or synthetic reward) | report actual value, no fixed target needed for portfolio |
| RAG grounding | % of answer claims traceable to a retrieved chunk (manual spot-check on 20 queries) | 100% for the demo-quality examples in the README |
| Fairness | Demographic parity difference across segments | < 10 percentage points, flagged if exceeded |
| API | p95 latency | < 1.5s for `/recommendations` without `query`, report actual for `/chat` |
| Engineering | Test coverage | report actual (aim for the core logic modules covered, not 100% of everything) |

---

## 13. Production Readiness Checklist

- [ ] `.env.example` committed, real `.env` gitignored — no secrets in the repo
- [ ] Every API input validated with Pydantic
- [ ] `/health` endpoint returns 200 when DB + vector store are reachable
- [ ] Structured logs (JSON) for every request, including latency and whether RAG
      context was found
- [ ] Errors reported to Sentry (or logged with stack trace at minimum)
- [ ] `docker-compose up` brings up a fully working stack from a clean clone
- [ ] CI runs lint + tests on every push and is green on `main`
- [ ] README documents: architecture diagram, how to run locally, how to run tests,
      the live demo link, and the "delete a doc → graceful degradation" example
- [ ] Fairness report artifact committed as an example output, not just described

---

## 14. Resume / Portfolio Framing

One line for a resume: *"Built and deployed a RAG-grounded, fairness-audited
hyper-personalization system for banking CRM — FastAPI + Postgres + Chroma, containerized,
CI/CD via GitHub Actions, live demo on [link]."*

For interviews, the two strongest talking points this project gives you:

1. **The grounding failure mode you designed against** — walk through the "delete a
   source document and the answer degrades instead of hallucinating" test. This is a
   concrete, demonstrable answer to "how do you prevent hallucination," which is one
   of the most common GenAI interview questions right now.
2. **The fairness audit as an engineering decision, not a feature request** — tie it
   back to your dissertation work on fairness control; this project shows you can
   carry that idea from research into a production-shaped system in a different
   domain.
