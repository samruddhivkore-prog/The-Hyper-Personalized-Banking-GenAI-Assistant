# Hyper-Personalized Banking GenAI Assistant

A backend service that segments bank customers, generates personalized product
recommendations with a transparent reason for each one, and answers free-text
customer questions grounded in real policy documents via RAG — with a fairness
audit running over every batch of recommendations before they go out.

Built end-to-end: FastAPI + Postgres + Chroma, containerized, tested (89%
coverage on `app/`), CI on every push. See [`hyper_personalized_banking_genai_spec.md`](hyper_personalized_banking_genai_spec.md)
for the original design spec this was built from.

## Architecture

```
                         ┌──────────────────────┐
                         │  Streamlit chat UI     │
                         └───────────┬────────────┘
                                     │ HTTPS
                         ┌───────────▼────────────┐
                         │       FastAPI app        │
                         │  /recommendations         │
                         │  /chat  /segments  /health │
                         └──┬─────────────────┬─────┘
              ┌─────────────┘                 └──────────────┐
   ┌──────────▼──────────┐              ┌──────────────────────▼─────────────┐
   │   Decisions layer     │              │           Design layer              │
   │ SegmentationModel      │              │ EmbeddingModel (sentence-transf.)   │
   │ RecommenderModel        │              │ VectorStore (Chroma)                 │
   │ FairnessAuditor          │              │ LLMClient (Groq / Gemini / Ollama /  │
   │                            │              │            stub for tests & CI)      │
   └──────────┬────────────┘              └──────────────────┬───────────────┘
              │                                                │
   ┌──────────▼────────────┐                       ┌──────────▼────────────┐
   │  Data Foundation         │                       │  Knowledge base docs    │
   │  Postgres (customers,     │                       │  data/kb_docs/*.md       │
   │  transactions, recs)        │                       │  (16 synthetic policies) │
   └────────────────────────┘                       └─────────────────────────┘
```

Design philosophy: segmentation and scoring are **deterministic classical ML**
(KMeans, rule-based scoring) — cheap, fast, auditable. The LLM is used only
where language is genuinely required: turning a recommendation list into
prose, and answering free-text policy questions. The LLM **never answers from
its own knowledge** — only from chunks retrieved at request time. If nothing
relevant is retrieved, or nothing relevant survives (e.g. the source document
was deleted), the answer degrades to admitting it, not guessing.

## Project layout

```
app/
  segmentation.py, recommender.py, fairness.py   — Decisions layer
  rag/                                             — Design layer (chunking, embeddings,
                                                      vector store, LLM client, prompts, pipeline)
  services/                                        — online + offline pipeline orchestration
  main.py, schemas.py, models.py, database.py      — FastAPI app, Pydantic/SQLAlchemy models
scripts/
  seed_db.py        — generate synthetic customers, load DB, run offline refresh
  index_kb.py        — chunk + embed + upsert the knowledge base
  offline_refresh.py — the scheduled batch job (segmentation, recs, fairness audit)
data/kb_docs/         — 16 synthetic bank policy documents (RAG source corpus)
frontend/              — Streamlit chat UI
tests/                  — 32 tests, 89% coverage on app/
reports/                — fairness report artifacts (fairness_sample_*.json committed as examples)
```

## Running locally

### Option A — Docker Compose (closest to production)

```bash
cp .env.example .env
docker compose up --build
docker compose exec api python -m scripts.seed_db --n 300
docker compose exec api python -m scripts.index_kb
```

- API: http://localhost:8000/docs
- Frontend: http://localhost:8501

### Option B — plain Python (SQLite, no Docker)

```bash
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m scripts.seed_db --n 300      # generates synthetic customers + runs offline refresh
python -m scripts.index_kb             # indexes the KB docs into the vector store
uvicorn app.main:app --reload
```

Without a `DATABASE_URL` set, the app falls back to a local SQLite file
(`banking_assistant.db`) automatically — no Postgres required for local dev.

> **Note on chromadb on Windows:** `chromadb`'s dependency `chroma-hnswlib`
> needs a C++ compiler to build from source, which a bare Windows install
> often lacks (prebuilt wheels exist for Linux, which is what the Docker
> image and CI use). If `chromadb` isn't importable, the app automatically
> falls back to a tiny in-memory cosine-similarity vector store
> (`app/rag/vector_store.py::InMemoryVectorStore`) so everything still runs —
> you just lose persistence across process restarts. Install MS C++ Build
> Tools, or use Docker, to get the real persistent Chroma store on Windows.

### Running tests

```bash
pip install -r requirements-dev.txt
pytest -v --cov=app --cov-report=term-missing
ruff check .
```

32 tests, 89% coverage on `app/`. Tests never hit a real LLM or a downloaded
embedding model: the LLM client defaults to `LLM_PROVIDER=stub` (a
deterministic, extractive, zero-network "LLM" — see below), and tests that
need embeddings force the hash-based fallback explicitly.

## The grounding test (why RAG, not a free-form LLM call)

The single most important correctness rule in the system: **the assistant
answers only from retrieved document chunks, never from what the model
"knows."** This is enforced twice — the prompt instructs it explicitly (see
`app/rag/prompts.py`), and the retrieval step returns nothing (triggering a
canned "I don't have enough information" answer) if no relevant chunk is
found.

Here's the actual captured output of the demonstration described in the
spec: query the assistant, then delete the one source document that grounds
the answer, and query again.

**Before** — `data/kb_docs/real_time_payments_faq.md` is indexed:

```json
{
  "answer": "... Real-Time Payments FAQ Real-Time Payments (RTP) lets you send
  money to another bank account and have it arrive in seconds, 24 hours a
  day, 7 days a week, including weekends and holidays. ...",
  "sources": ["ACH Transfer Policy", "Wire Transfer Fees", "Real-Time Payments FAQ"],
  "confidence": "medium"
}
```

**After** deleting that document from the vector store
(`python -m scripts.index_kb --remove real_time_payments_faq`):

```json
{
  "answer": "... Choose Real-Time Payments when the recipient needs funds
  within seconds, or a wire transfer for same-day delivery of large amounts
  above $5,000. ...",
  "sources": ["ACH Transfer Policy", "Wire Transfer Fees", "Overdraft Protection"],
  "confidence": "medium"
}
```

The $1.00 fee and "seconds" claim disappear along with the document — the
model stops asserting facts it can no longer see, instead of inventing them.
`tests/test_rag_pipeline.py::test_answer_query_degrades_gracefully_when_doc_deleted`
covers the fully-empty-retrieval case explicitly (answer becomes exactly *"I
don't have enough information to answer that."*).

## The fairness auditor — an honest example, not a cherry-picked one

The auditor computes demographic parity (max − min recommendation rate) per
product across groups defined by **protected-adjacent attributes** — age
bracket and income bracket — not the KMeans customer segment. (Segment
membership is itself *derived from* age/tenure/channel/income, so grouping by
segment flags nearly every product as an artifact of clustering, not unfair
treatment — an early version of this project did exactly that; see
`app/fairness.py`'s module docstring.)

Some parity differences are expected because a product has an explicit
eligibility gate on that attribute (`personal_loan` and `rewards_credit_card`
require a minimum income bracket; `retirement_ira` and
`student_loan_refinance` gate on age). Those are recorded and still shown in
the full report for transparency, but are excluded from the per-request
`fairness_flags` returned to callers, which is reserved for differences with
no such justification.

Running the audit against 150 synthetic customers (`reports/fairness_sample_income_bracket.json`,
`reports/fairness_sample_age_bracket.json`, committed as example artifacts)
surfaced a real, unintentional finding: `high_yield_savings`, `real_time_payments`,
and `retirement_ira` use income/tenure as a **continuous multiplier** in their
scoring functions rather than a hard eligibility gate (see
`app/recommender.py`), producing a 25-34 percentage-point gap across income
brackets with no documented justification. That's the fairness auditor doing
its job — catching a soft bias the recommender's author (me) didn't
intentionally design and wouldn't have noticed without it. Fixing the
underlying scoring functions is a natural next step, deliberately left as-is
here so the report artifact demonstrates a real catch rather than a sanitized
example.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | DB + vector store reachability |
| `/recommendations` | POST | `{customer_id, query?}` → recommendations + grounded answer + fairness flags |
| `/chat` | POST | Same contract as `/recommendations`, `query` required |
| `/segments` | GET | Segment id, label, and customer count per segment |

Full request/response schemas: `app/schemas.py`, or `/docs` (Swagger UI) once
the app is running.

## Evaluation metrics (actual, not targets)

| Layer | Metric | Result |
|---|---|---|
| Segmentation | Silhouette score, k=5, n=150 synthetic customers | 0.20 |
| RAG grounding | Manual spot-check, 5 queries against the KB | No claims traceable outside retrieved chunks |
| Fairness | Demographic parity, threshold 10pp | 5/8 products flagged as unjustified per dimension on synthetic data — see above |
| Test coverage | `pytest --cov=app` | 89% |
| API | p95 latency, `/recommendations` without `query` | Not load-tested; single-request latency ~5ms against SQLite locally |

The 0.20 silhouette score is below the spec's 0.3 sanity-floor suggestion.
With only 6 lightly-correlated features and k=5 on synthetic data with no
truly separated clusters, this is expected rather than a bug — a real
Kaggle/TabFormer dataset with richer, genuinely correlated features would
likely score higher. Documented honestly rather than tuned to hit a number.

## Deployment

This repository is deploy-ready but **has not been deployed** — that requires
accounts and credentials (Render/Fly.io, Neon/Supabase, Hugging Face Spaces,
a Groq or Gemini API key) that weren't available in the environment this was
built in. To deploy:

1. **Database**: create a free Postgres instance (Neon or Supabase), set
   `DATABASE_URL` to its connection string.
2. **API**: push this repo to GitHub, connect it to Render or Fly.io as a
   Docker deploy, set env vars from `.env.example` (in particular
   `LLM_PROVIDER=groq` and `GROQ_API_KEY`, or `gemini`/`GEMINI_API_KEY` — both
   have free tiers), point `DATABASE_URL` at the hosted Postgres, and mount a
   persistent volume at `CHROMA_PERSIST_DIR` (or move to Chroma Cloud /
   re-run `scripts/index_kb.py` on boot).
3. **Frontend**: deploy `Dockerfile.frontend` to a Hugging Face Space, set
   `API_BASE_URL` to the deployed API's public URL.
4. Run `python -m scripts.seed_db` and `python -m scripts.index_kb` once
   against the deployed database to populate it.
5. Confirm `/health` returns `{"database": true, "vector_store": true}` and
   run the grounding test above against the live URL for the README/demo.

## Production readiness checklist

- [x] `.env.example` committed, real `.env` gitignored
- [x] Every API input validated with Pydantic
- [x] `/health` returns DB + vector store reachability
- [x] Structured JSON logs per request (latency, whether RAG context was found)
- [x] Sentry hook wired (`SENTRY_DSN` env var; no-op if unset)
- [x] `docker-compose up` brings up API + Postgres + frontend from a clean clone
- [x] CI runs lint + tests on every push (`.github/workflows/ci.yml`)
- [x] README documents architecture, local run, tests, and the grounding-degradation example
- [x] Fairness report artifacts committed as real examples (`reports/fairness_sample_*.json`)
- [ ] Live demo link — not deployed (see Deployment above)

