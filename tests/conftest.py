import os
import tempfile

os.environ.setdefault("LLM_PROVIDER", "stub")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault(
    "CHROMA_PERSIST_DIR", os.path.join(tempfile.gettempdir(), "banking_assistant_test_chroma")
)

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database import Base, get_db
from app.main import app
from app.models import Customer

TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(bind=TEST_ENGINE, autoflush=False, autocommit=False)


@pytest.fixture()
def db_session():
    Base.metadata.create_all(bind=TEST_ENGINE)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture()
def client(db_session):
    def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def sample_profile() -> dict:
    return {
        "customer_id": "CUST00001",
        "age": 29,
        "income_bracket": "50k_75k",
        "account_tenure_months": 34,
        "products_held": ["checking", "savings"],
        "channel_preference": "mobile",
        "avg_monthly_transactions": 42,
    }


@pytest.fixture()
def seeded_customer(db_session, sample_profile) -> Customer:
    customer = Customer(
        customer_id=sample_profile["customer_id"],
        age=sample_profile["age"],
        income_bracket=sample_profile["income_bracket"],
        account_tenure_months=sample_profile["account_tenure_months"],
        products_held=sample_profile["products_held"],
        channel_preference=sample_profile["channel_preference"],
        avg_monthly_transactions=sample_profile["avg_monthly_transactions"],
        segment_id=0,
        segment_label="digital_first_young_professional",
    )
    db_session.add(customer)
    db_session.commit()
    return customer
