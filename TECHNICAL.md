# Technical README

This is the engineering-depth companion to the main [README.md](README.md),
which is written for a non-technical reader. Everything below assumes
familiarity with the usual web/ML stack terms.

> **Note:** this file was written earlier in the project's life and some
> counts (e.g. "16 docs") are stale — the knowledge base now has 19 policy
> documents plus spending analysis, a card-freeze action flow, a product
> catalog for "which product fits me" questions, and a separate staff admin
> app. The architecture description, grounding/fairness discussion, and
> local-run instructions below are still accurate; treat specific numbers as
> a snapshot rather than current totals.

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
   │  transactions, recs)        │                       │  (synthetic policies)    │
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
  services/                                        — online + offline pipeline orchestration,
                                                      spending analysis, card actions, product
                                                      matching, question-type classification
  main.py, schemas.py, models.py, database.py      — FastAPI app, Pydantic/SQLAlchemy models
scripts/
  seed_db.py        — generate synthetic customers, load DB, run offline refresh
  index_kb.py        — chunk + embed + upsert the knowledge base
  offline_refresh.py — the scheduled batch job (segmentation, recs, fairness audit)
data/kb_docs/         — synthetic bank policy documents (RAG source corpus)
data/product_catalog.json — structured product metadata for "which product fits me" questions
frontend/
  streamlit_app.py    — customer-facing chat app
  admin_app.py         — staff-only console (segments, scores, fairness report, action log)
tests/                  — unit + integration tests
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
- Customer app: http://localhost:8501
- Admin app: run `streamlit run frontend/admin_app.py --server.port 8502` separately
  (set `ADMIN_PASSWORD` to gate it; without it, the app opens with a warning banner)

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

The API also auto-indexes the KB on startup if the vector store is empty, so
skipping the manual `index_kb` step still works — this matters because the
in-memory fallback store (used automatically when `chromadb` isn't
installed) doesn't persist across processes, so a separately-run indexing
script would otherwise never reach the running API process.

> **Note on chromadb on Windows:** `chromadb`'s dependency `chroma-hnswlib`
> needs a C++ compiler to build from source, which a bare Windows install
> often lacks (prebuilt wheels exist for Linux, which is what the Docker
> image and CI use). It's kept in `requirements-chroma.txt`, separate from
> `requirements.txt`, so a Windows dev machine without build tools can skip
> it — the app falls back to a tiny in-memory cosine-similarity vector store
> (`app/rag/vector_store.py::InMemoryVectorStore`) automatically, so
> everything still runs; you just lose persistence across process restarts.
> Install MS C++ Build Tools, `pip install -r requirements-chroma.txt`, or
> use Docker, to get the real persistent Chroma store on Windows.

### Running tests

```bash
pip install -r requirements-dev.txt
pytest -v --cov=app --cov-report=term-missing
ruff check .
```

Tests never hit a real LLM or a downloaded embedding model by default: the
LLM client falls back to `LLM_PROVIDER=stub` (a deterministic, extractive,
zero-network "LLM") whenever no provider API key is set, and tests that need
embeddings force the hash-based fallback explicitly. Setting `GROQ_API_KEY`
(or `GEMINI_API_KEY`/`ANTHROPIC_API_KEY`) un-skips the handful of tests that
need real model judgement (e.g. telling a mortgage question apart from a
personal loan one).

## The grounding test (why RAG, not a free-form LLM call)

The single most important correctness rule in the system: **the assistant
answers only from retrieved document chunks, never from what the model
"knows."** This is enforced three ways: the prompt instructs it explicitly
(see `app/rag/prompts.py`); retrieval returning nothing triggers the fallback
message instead of a guess; and — because embedding similarity scores alone
aren't a reliable relevance signal, especially for the hash-based fallback
embedding used when `chromadb`/`sentence-transformers` aren't available — a
keyword-overlap relevance gate (`app/rag/relevance.py`) rejects retrieved
chunks that don't actually share any meaningful word with the question, even
if something was technically retrieved. A small explicit veto list also
exists for product categories the bank deliberately never offers (insurance,
crypto) — see `app/rag/pipeline.py::OUT_OF_SCOPE_TERMS` — since those can
share an incidental word with an unrelated real product. All paths return
the same customer-facing fallback: *"I couldn't find this in our policies.
Please contact the bank directly."*

Delete the one source document that grounds a query
(`python -m scripts.index_kb --remove real_time_payments_faq`) and ask the
same question again: the answer stops mentioning that product's facts
entirely and falls back to whatever's still grounded — it never invents the
deleted product's numbers. `tests/test_rag_pipeline.py` and
`tests/test_grounding_quality.py` cover this mechanically.

All currency in the knowledge base, API examples, and UI is EUR (the bank is
based in Ireland) — consistently, not a mix of $ and €. US-only terminology
(ACH, "check", IRA) was replaced with Irish/EU equivalents: SEPA Credit
Transfer / SEPA Instant for transfers, "cheque" for cheques, PRSA for the
pension product.

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
require a minimum income bracket; `pension_prsa` and `student_loan_refinance`
gate on age). Those are recorded and still shown in the full report for
transparency, but are excluded from the per-request `fairness_flags` returned
to callers, which is reserved for differences with no such justification.

Running the audit against synthetic customers (`reports/fairness_sample_*.json`,
committed as example artifacts) surfaced a real, unintentional finding:
`high_yield_savings` and `real_time_payments` use income/tenure as a
**continuous multiplier** in their scoring functions rather than a hard
eligibility gate, producing a double-digit percentage-point gap across income
brackets with no documented justification. That's the fairness auditor doing
its job — catching a soft bias the recommender's author didn't intentionally
design and wouldn't have noticed without it. Fixing the underlying scoring
functions is a natural next step, deliberately left as-is here so the report
artifact demonstrates a real catch rather than a sanitized example.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | DB + vector store reachability |
| `/recommendations` | POST | `{customer_id, query?}` → recommendations + grounded answer + fairness flags |
| `/chat` | POST | Same contract as `/recommendations`, routes to spending/card-action/product-match/policy handling |
| `/segments` | GET | Segment id, label, and customer count per segment |
| `/customers/{id}/spending` | GET | Spend total, category breakdown, and change vs. the previous period |
| `/actions/freeze-card`, `/actions/unfreeze-card` | POST | Card status change — only ever called after explicit customer confirmation |
| `/admin/status`, `/admin/fairness`, `/admin/actions-log` | GET | Staff-only data, consumed by `frontend/admin_app.py` |

Each recommendation carries both the internal `score`/`reason_code` (admin)
and a customer-facing `plain_text`/`learn_more` sentence. Each assistant
response carries both `confidence`/`retrieved_chunks` (admin) and the plain
`answer`/`sources` customers see, plus optional `pending_action` (a card
action awaiting Yes/No) and `spending_data` (structured numbers for a chart).
The API always returns all of these fields — hiding the admin ones is a
frontend concern, not a backend one, so other clients can still use them.

Full request/response schemas: `app/schemas.py`, or `/docs` (Swagger UI) once
the app is running.

## Evaluation metrics (actual, not targets)

| Layer | Metric | Result |
|---|---|---|
| Segmentation | Silhouette score, k=5, n=150 synthetic customers | 0.20 |
| RAG grounding | Manual spot-check across dozens of queries | No claims traceable outside retrieved chunks |
| Fairness | Demographic parity, threshold 10pp | Several products flagged as unjustified per dimension on synthetic data — see above |
| Test coverage | `pytest --cov=app` | ~89% on `app/` |

The 0.20 silhouette score is below the spec's 0.3 sanity-floor suggestion.
With only 6 lightly-correlated features and k=5 on synthetic data with no
truly separated clusters, this is expected rather than a bug — a real
Kaggle/TabFormer dataset with richer, genuinely correlated features would
likely score higher. Documented honestly rather than tuned to hit a number.

## Known limitations

- **Ambiguous product questions sometimes need a real AI model, not the free
  built-in fallback.** Telling "a personal loan is not a mortgage" apart, or
  rejecting "car insurance" even though a loan doc happens to mention
  "buying a car," needs actual judgement — the keyword/score heuristics
  alone got this wrong in both directions at various points during
  development, and a small number of these cases are only reliably correct
  with `LLM_PROVIDER` pointed at a real provider.
- **Not deployed live.** Deploy-ready (Docker, CI, env-driven config) but no
  hosting account was available in the environment this was built in.
- **Free-tier model names drift.** Groq in particular has retired and
  replaced hosted model IDs during this project's lifetime; `GROQ_MODEL` /
  `LLM_MODEL` exist specifically so this doesn't require a code change.

## Deployment

This repository is deploy-ready but **has not been deployed** — that requires
accounts and credentials (Render/Fly.io, Neon/Supabase, Hugging Face Spaces,
a Groq/Gemini/Anthropic API key) that weren't available in the environment
this was built in. To deploy:

1. **Database**: create a free Postgres instance (Neon or Supabase), set
   `DATABASE_URL` to its connection string.
2. **API**: push this repo to GitHub, connect it to Render or Fly.io as a
   Docker deploy, set env vars from `.env.example` (in particular
   `LLM_PROVIDER` and the matching API key — all three real providers have
   free tiers), point `DATABASE_URL` at the hosted Postgres, and mount a
   persistent volume at `CHROMA_PERSIST_DIR` (or move to Chroma Cloud /
   re-run `scripts/index_kb.py` on boot).
3. **Frontend**: deploy `Dockerfile.frontend` to a Hugging Face Space, set
   `API_BASE_URL` to the deployed API's public URL. Deploy the admin app
   separately with `ADMIN_PASSWORD` set.
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

## Resume line

*Built and (locally validated, deploy-ready) a RAG-grounded, fairness-audited
hyper-personalization system for banking CRM — FastAPI + Postgres + Chroma,
multi-provider LLM support, containerized, CI/CD via GitHub Actions.*
