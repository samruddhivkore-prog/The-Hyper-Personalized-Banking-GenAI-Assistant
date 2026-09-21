from app.recommender import RecommenderModel


def _profile(**overrides):
    base = {
        "customer_id": "CUST1",
        "age": 29,
        "income_bracket": "50k_75k",
        "account_tenure_months": 34,
        "products_held": ["checking"],
        "channel_preference": "mobile",
        "avg_monthly_transactions": 42,
    }
    base.update(overrides)
    return base


def test_every_recommendation_has_a_reason_code():
    recommender = RecommenderModel()
    results = recommender.score(_profile())
    assert results
    for r in results:
        assert r["reason_code"]
        assert 0 <= r["score"] <= 1


def test_high_transfer_frequency_triggers_real_time_payments():
    recommender = RecommenderModel()
    results = recommender.score(_profile(avg_monthly_transactions=50))
    products = {r["product_id"]: r for r in results}
    assert "real_time_payments" in products
    assert products["real_time_payments"]["reason_code"] == "high_transfer_frequency"


def test_low_transfer_frequency_excludes_real_time_payments():
    recommender = RecommenderModel()
    results = recommender.score(_profile(avg_monthly_transactions=3))
    products = {r["product_id"] for r in results}
    assert "real_time_payments" not in products


def test_already_held_product_is_not_recommended_again():
    recommender = RecommenderModel()
    results = recommender.score(_profile(products_held=["checking", "high_yield_savings"]))
    products = {r["product_id"] for r in results}
    assert "high_yield_savings" not in products


def test_new_account_not_eligible_for_personal_loan():
    recommender = RecommenderModel()
    results = recommender.score(_profile(account_tenure_months=2))
    products = {r["product_id"] for r in results}
    assert "personal_loan" not in products


def test_results_sorted_descending_by_score():
    recommender = RecommenderModel()
    results = recommender.score(_profile(avg_monthly_transactions=60, age=45, account_tenure_months=100))
    scores = [r["score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_limits_result_count():
    recommender = RecommenderModel()
    results = recommender.score(_profile(avg_monthly_transactions=60), top_k=2)
    assert len(results) <= 2
