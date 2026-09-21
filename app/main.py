from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db, init_db
from app.logging_config import configure_logging, get_logger
from app.models import Customer
from app.schemas import (
    ChatRequest,
    HealthResponse,
    RecommendationRequest,
    RecommendationResponse,
    SegmentOut,
)
from app.services.recommendation_service import handle_recommendation_request

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    if settings.sentry_dsn:
        import sentry_sdk

        sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.1)
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
