from datetime import date, datetime, timedelta

from app.models import Card, Transaction
from app.services.period_parser import parse_period
from app.services.spending import compute_spending_summary


def _add_txn(db, customer_id, days_ago, category, amount):
    db.add(
        Transaction(
            customer_id=customer_id,
            date=datetime.now() - timedelta(days=days_ago),
            category=category,
            amount=amount,
        )
    )


# ---- TASK 7 #4: period parser ----


def test_period_parser_last_month():
    today = date(2026, 3, 15)
    p = parse_period("last month", today=today)
    assert p.start == date(2026, 2, 1)
    assert p.end == date(2026, 2, 28)


def test_period_parser_named_month_current_year():
    today = date(2026, 10, 5)
    p = parse_period("what did I spend in August", today=today)
    assert p.start == date(2026, 8, 1)
    assert p.end == date(2026, 8, 31)


def test_period_parser_named_month_with_year():
    p = parse_period("March 2026")
    assert p.start == date(2026, 3, 1)
    assert p.end == date(2026, 3, 31)


def test_period_parser_last_3_months():
    today = date(2026, 10, 5)
    p = parse_period("last 3 months", today=today)
    assert p.start == date(2026, 7, 1)
    assert p.end == date(2026, 9, 30)


def test_period_parser_n_days_ago():
    today = date(2026, 10, 5)
    p = parse_period("120 days ago", today=today)
    assert p.end == today
    assert (p.end - p.start).days == 119


def test_period_parser_rejects_nonsense():
    assert parse_period("asdkjfh qwoeiru zzzz") is None


# ---- TASK 7 #3: spending totals, percentages, change, empty period ----


def test_spending_summary_30_day_range(db_session, seeded_customer):
    today = datetime.now()
    _add_txn(db_session, seeded_customer.customer_id, 5, "groceries", 100.0)
    _add_txn(db_session, seeded_customer.customer_id, 10, "groceries", 50.0)
    _add_txn(db_session, seeded_customer.customer_id, 15, "transport", 50.0)
    db_session.commit()

    from app.services.period_parser import Period

    period = Period(today.date() - timedelta(days=29), today.date(), "the last 30 days")
    summary = compute_spending_summary(db_session, seeded_customer.customer_id, period)

    assert summary.total == 200.0
    assert summary.has_data
    by_cat = {c.category: c for c in summary.categories}
    assert by_cat["groceries"].amount == 150.0
    assert round(by_cat["groceries"].pct) == 75
    assert round(by_cat["transport"].pct) == 25
    assert summary.top_categories[0] == "groceries"


def test_spending_summary_120_day_range_and_change_vs_previous(db_session, seeded_customer):
    today = datetime.now()
    # recent period: days 0-119
    _add_txn(db_session, seeded_customer.customer_id, 10, "bills", 100.0)
    # previous period of equal length: days 120-239
    _add_txn(db_session, seeded_customer.customer_id, 130, "bills", 50.0)
    db_session.commit()

    from app.services.period_parser import Period

    period = Period(today.date() - timedelta(days=119), today.date(), "the last 120 days")
    summary = compute_spending_summary(db_session, seeded_customer.customer_id, period)

    assert summary.total == 100.0
    assert summary.previous_total == 50.0
    assert summary.change_pct == 100.0  # spent double vs the prior equal-length period


def test_spending_summary_named_month(db_session, seeded_customer):
    db_session.add(
        Transaction(
            customer_id=seeded_customer.customer_id,
            date=datetime(2026, 8, 10),
            category="dining",
            amount=60.0,
        )
    )
    db_session.commit()

    from app.services.period_parser import Period

    period = Period(date(2026, 8, 1), date(2026, 8, 31), "August 2026")
    summary = compute_spending_summary(db_session, seeded_customer.customer_id, period)
    assert summary.total == 60.0
    assert summary.categories[0].category == "dining"


def test_spending_summary_empty_period_returns_no_data(db_session, seeded_customer):
    from app.services.period_parser import Period

    period = Period(date(2020, 1, 1), date(2020, 1, 31), "January 2020")
    summary = compute_spending_summary(db_session, seeded_customer.customer_id, period)
    assert summary.has_data is False
    assert summary.total == 0.0


# ---- TASK 7 #5: chat spending questions use data, not policy search ----


def test_chat_spending_question_uses_spending_data_not_policy(client, seeded_customer, db_session):
    _add_txn(db_session, seeded_customer.customer_id, 5, "groceries", 80.0)
    db_session.commit()

    resp = client.post(
        "/chat",
        json={
            "customer_id": seeded_customer.customer_id,
            "query": "how much did I spend on groceries last month",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["assistant_response"]["spending_data"] is not None
    assert body["assistant_response"]["sources"] == []


# ---- TASK 7 #6 and #7: freeze flow, confirmation, ownership ----


def _freeze(client, customer_id, card_id):
    return client.post("/actions/freeze-card", json={"customer_id": customer_id, "card_id": card_id})


def test_freeze_requires_no_db_change_before_confirmation(client, seeded_customer, seeded_card, db_session):
    resp = client.post("/chat", json={"customer_id": seeded_customer.customer_id, "query": "freeze my card"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["assistant_response"]["pending_action"]["action"] == "freeze_card"
    assert body["assistant_response"]["pending_action"]["last4"] == "4821"

    db_session.expire_all()
    card = db_session.get(Card, seeded_card.card_id)
    assert card.status == "active"  # unchanged — no confirmation clicked yet


def test_freeze_confirmation_endpoint_changes_status_once(client, seeded_customer, seeded_card, db_session):
    resp = _freeze(client, seeded_customer.customer_id, seeded_card.card_id)
    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    assert resp.json()["status"] == "frozen"

    db_session.expire_all()
    assert db_session.get(Card, seeded_card.card_id).status == "frozen"

    # second freeze says already frozen, no error
    resp2 = _freeze(client, seeded_customer.customer_id, seeded_card.card_id)
    assert "already frozen" in resp2.json()["message"].lower()


def test_customer_cannot_freeze_another_customers_card(client, seeded_customer, second_card):
    resp = _freeze(client, seeded_customer.customer_id, second_card.card_id)
    body = resp.json()
    assert body["ok"] is False
    assert second_card.status == "active"
