from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    age: Mapped[int] = mapped_column(Integer)
    income_bracket: Mapped[str] = mapped_column(String(32))
    account_tenure_months: Mapped[int] = mapped_column(Integer)
    products_held: Mapped[list] = mapped_column(JSON, default=list)
    channel_preference: Mapped[str] = mapped_column(String(32))
    avg_monthly_transactions: Mapped[float] = mapped_column(Float)
    segment_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    segment_label: Mapped[str | None] = mapped_column(String(64), nullable=True)

    transactions: Mapped[list["Transaction"]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )
    recommendations: Mapped[list["Recommendation"]] = relationship(
        back_populates="customer", cascade="all, delete-orphan"
    )

    def features(self) -> dict:
        return {
            "age": self.age,
            "income_bracket": self.income_bracket,
            "account_tenure_months": self.account_tenure_months,
            "channel_preference": self.channel_preference,
            "avg_monthly_transactions": self.avg_monthly_transactions,
            "num_products_held": len(self.products_held or []),
        }


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    date: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
    category: Mapped[str] = mapped_column(String(64))
    amount: Mapped[float] = mapped_column(Float)

    customer: Mapped["Customer"] = relationship(back_populates="transactions")


class Product(Base):
    __tablename__ = "products"

    product_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    category: Mapped[str] = mapped_column(String(64))
    eligibility_rule: Mapped[str] = mapped_column(String(256), default="")
    description: Mapped[str] = mapped_column(String(512), default="")


class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(ForeignKey("customers.customer_id"))
    product_id: Mapped[str] = mapped_column(String(64))
    product_name: Mapped[str] = mapped_column(String(128))
    score: Mapped[float] = mapped_column(Float)
    reason_code: Mapped[str] = mapped_column(String(64))
    segment_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    customer: Mapped["Customer"] = relationship(back_populates="recommendations")


class InteractionLog(Base):
    __tablename__ = "interaction_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(String(32))
    query: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    response_json: Mapped[dict] = mapped_column(JSON)
    latency_ms: Mapped[float] = mapped_column(Float)
    rag_context_found: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)
