from contextlib import asynccontextmanager
from datetime import date

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, init_db
from app.logging_config import configure_logging, get_logger
from app.models import ActionLog, Customer
from app.schemas import (
    ActionLogOut,
    AdminStatusOut,
    CardActionRequest,
    CardActionResponse,
    ChatRequest,
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
    SegmentOut,
    SpendingResponse,
)
from app.services.cards import freeze_card, unfreeze_card
from app.services.period_parser import Period, default_period
from app.services.recommendation_service import handle_recommendation_request
from app.services.spending import compute_spending_summary, render_spending_sentence

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    if settings.sentry_dsn:
        import sentry_sdk

        sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.1)

    from app.rag.llm_client import describe_llm_provider

    logger.info("llm_provider_selected", provider=f"LLM: {describe_llm_provider()}")

    # Without chromadb, the vector store falls back to an in-memory one that
    # doesn't persist across processes — if `scripts/index_kb.py` was run
    # separately, this process would otherwise start with an empty store and
    # every question would silently get zero retrieval. Index here if empty
    # so the app is correct out of the box regardless of backend.
    try:
        from app.rag.vector_store import get_vector_store

        if get_vector_store().count() == 0:
            from scripts.index_kb import index_kb

            index_kb()
    except Exception:
        logger.warning("startup_kb_index_failed", exc_info=True)

    yield


app = FastAPI(title="Hyper-Personalized Banking Assistant", version="1.0.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(db: Session = Depends(get_db)):
    db_ok = True
    try:
        db.execute(text("SELECT 1"))
    except Exception:
        db_ok = False

    vs_ok = True
    try:
        from app.rag.vector_store import get_vector_store

        get_vector_store()
    except Exception:
        vs_ok = False

    status = "ok" if db_ok and vs_ok else "degraded"
    return HealthResponse(status=status, database=db_ok, vector_store=vs_ok)


@app.post("/recommendations", response_model=RecommendationResponse)
def get_recommendations(req: RecommendationRequest, db: Session = Depends(get_db)):
    result = handle_recommendation_request(db, req.customer_id, req.query)
    if result is None:
        raise HTTPException(status_code=404, detail="customer not found")
    return result


@app.post("/chat", response_model=RecommendationResponse)
def chat(req: ChatRequest, db: Session = Depends(get_db)):
    result = handle_recommendation_request(db, req.customer_id, req.query)
    if result is None:
        raise HTTPException(status_code=404, detail="customer not found")
    return result


@app.get("/segments", response_model=list[SegmentOut])
def list_segments(db: Session = Depends(get_db)):
    rows = (
        db.query(Customer.segment_id, Customer.segment_label, func.count(Customer.customer_id))
        .filter(Customer.segment_id.isnot(None))
        .group_by(Customer.segment_id, Customer.segment_label)
        .all()
    )
    return [
        SegmentOut(segment_id=seg_id, label=label or "unlabeled", customer_count=count)
        for seg_id, label, count in rows
    ]


@app.get("/customers/{customer_id}/spending", response_model=SpendingResponse)
def get_spending(
    customer_id: str,
    start_date: date | None = None,
    end_date: date | None = None,
    month: str | None = None,
    db: Session = Depends(get_db),
):
    if db.get(Customer, customer_id) is None:
        raise HTTPException(status_code=404, detail="customer not found")

    if month:
        import calendar

        try:
            year, mon = int(month.split("-")[0]), int(month.split("-")[1])
            start = date(year, mon, 1)
        except (ValueError, IndexError):
            raise HTTPException(status_code=422, detail="month must be YYYY-MM") from None
        end = date(year, mon, calendar.monthrange(year, mon)[1])
        period = Period(start, end, start.strftime("%B %Y"))
    elif start_date and end_date:
        period = Period(start_date, end_date, f"{start_date.isoformat()} to {end_date.isoformat()}")
    else:
        period = default_period()

    summary = compute_spending_summary(db, customer_id, period)
    sentence = render_spending_sentence(summary)
    return SpendingResponse(customer_id=customer_id, sentence=sentence, **summary.to_dict())


@app.post("/actions/freeze-card", response_model=CardActionResponse)
def post_freeze_card(req: CardActionRequest, db: Session = Depends(get_db)):
    if db.get(Customer, req.customer_id) is None:
        raise HTTPException(status_code=404, detail="customer not found")
    result = freeze_card(db, req.customer_id, req.card_id)
    return CardActionResponse(ok=result.ok, message=result.message, status=result.status)


@app.post("/actions/unfreeze-card", response_model=CardActionResponse)
def post_unfreeze_card(req: CardActionRequest, db: Session = Depends(get_db)):
    if db.get(Customer, req.customer_id) is None:
        raise HTTPException(status_code=404, detail="customer not found")
    result = unfreeze_card(db, req.customer_id, req.card_id)
    return CardActionResponse(ok=result.ok, message=result.message, status=result.status)


@app.get("/admin/actions-log", response_model=list[ActionLogOut])
def get_actions_log(db: Session = Depends(get_db)):
    rows = db.query(ActionLog).order_by(ActionLog.created_at.desc()).limit(200).all()
    return [
        ActionLogOut(
            customer_id=r.customer_id,
            action=r.action,
            status=r.status,
            details=r.details,
            created_at=r.created_at,
        )
        for r in rows
    ]


@app.get("/admin/fairness")
def get_admin_fairness():
    from app.fairness import load_latest_reports

    return [r.to_dict() for r in load_latest_reports()]


@app.get("/admin/status", response_model=AdminStatusOut)
def get_admin_status():
    from app.rag.llm_client import describe_llm_provider

    return AdminStatusOut(llm_provider=describe_llm_provider())
