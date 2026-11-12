from datetime import datetime

from pydantic import BaseModel, Field


class TransactionIn(BaseModel):
    date: str
    category: str
    amount: float


class CustomerIn(BaseModel):
    customer_id: str
    age: int
    income_bracket: str
    account_tenure_months: int
    products_held: list[str] = Field(default_factory=list)
    channel_preference: str
    avg_monthly_transactions: float
    recent_transactions: list[TransactionIn] = Field(default_factory=list)


class RecommendationRequest(BaseModel):
    customer_id: str
    query: str | None = None


class RecommendationOut(BaseModel):
    product: str
    score: float
    reason_code: str
    plain_text: str  # customer-facing sentence — see app/services/copy.py
    learn_more: str  # customer-facing product blurb — see app/services/copy.py


class PendingAction(BaseModel):
    action: str  # "freeze_card" | "unfreeze_card"
    card_id: str
    last4: str
    message: str  # e.g. "Freeze your card ending 4821?"


class CategoryAmount(BaseModel):
    category: str
    amount: float
    pct: float


class SpendingData(BaseModel):
    period_label: str
    start: str
    end: str
    total: float
    categories: list[CategoryAmount] = Field(default_factory=list)
    top_categories: list[str] = Field(default_factory=list)
    previous_total: float | None = None
    change_pct: float | None = None
    has_data: bool = True


class AssistantResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: str  # "high" | "medium" | "none"
    retrieved_chunks: list[str] = Field(default_factory=list)  # admin-only debugging info
    pending_action: PendingAction | None = None  # set when a card action needs Yes/No confirmation
    spending_data: SpendingData | None = None  # set when the question was a spending lookup


class RecommendationResponse(BaseModel):
    customer_id: str
    segment: str | None
    recommendations: list[RecommendationOut]
    assistant_response: AssistantResponse
    fairness_flags: list[str] = Field(default_factory=list)
    timestamp: datetime


class ChatRequest(BaseModel):
    customer_id: str
    query: str


class SegmentOut(BaseModel):
    segment_id: int
    label: str
    customer_count: int


class HealthResponse(BaseModel):
    status: str
    database: bool
    vector_store: bool


class SpendingResponse(BaseModel):
    customer_id: str
    period_label: str
    start: str
    end: str
    total: float
    categories: list[CategoryAmount] = Field(default_factory=list)
    top_categories: list[str] = Field(default_factory=list)
    previous_total: float | None = None
    change_pct: float | None = None
    has_data: bool
    sentence: str


class CardActionRequest(BaseModel):
    customer_id: str
    card_id: str


class CardActionResponse(BaseModel):
    ok: bool
    message: str
    status: str | None = None


class ActionLogOut(BaseModel):
    customer_id: str
    action: str
    status: str
    details: str
    created_at: datetime


class AdminStatusOut(BaseModel):
    llm_provider: str
