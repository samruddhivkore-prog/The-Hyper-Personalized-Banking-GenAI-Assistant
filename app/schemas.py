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


class AssistantResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)
    confidence: str  # "high" | "medium" | "low" | "none"


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
