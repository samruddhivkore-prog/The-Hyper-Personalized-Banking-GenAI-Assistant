from app.segmentation import SegmentationModel, encode_features


def _profile(customer_id, **overrides):
    base = {
        "customer_id": customer_id,
        "age": 30,
        "income_bracket": "50k_75k",
        "account_tenure_months": 24,
        "products_held": ["checking"],
        "channel_preference": "mobile",
        "avg_monthly_transactions": 20,
    }
    base.update(overrides)
    return base


def test_encode_features_shape():
    profiles = [_profile("A"), _profile("B", age=60)]
    df = encode_features(profiles)
    assert list(df.index) == ["A", "B"]
    assert "age" in df.columns
    assert "channel_digital" in df.columns


def test_fit_assigns_every_customer_a_segment():
    profiles = [_profile(f"C{i}", age=25 + i, avg_monthly_transactions=10 + i) for i in range(10)]
    model = SegmentationModel(k=3)
    result = model.fit(profiles)

    assert set(result.assignments.keys()) == {p["customer_id"] for p in profiles}
    assert all(0 <= seg < 3 for seg in result.assignments.values())
    assert set(result.labels.keys()) == set(result.assignments.values())


def test_fit_produces_reasonable_silhouette_on_separated_clusters():
    young_digital = [
        _profile(f"Y{i}", age=25, avg_monthly_transactions=50, channel_preference="mobile")
        for i in range(15)
    ]
    senior_branch = [
        _profile(f"S{i}", age=65, avg_monthly_transactions=4, channel_preference="branch")
        for i in range(15)
    ]
    model = SegmentationModel(k=2)
    result = model.fit(young_digital + senior_branch)

    assert result.silhouette is not None
    assert result.silhouette > 0.3

    seg_young = result.assignments["Y0"]
    seg_senior = result.assignments["S0"]
    assert seg_young != seg_senior
    assert all(result.assignments[f"Y{i}"] == seg_young for i in range(15))
    assert all(result.assignments[f"S{i}"] == seg_senior for i in range(15))


def test_labels_are_human_readable_strings():
    profiles = [_profile(f"C{i}", age=25 + i) for i in range(6)]
    model = SegmentationModel(k=2)
    result = model.fit(profiles)
    for label in result.labels.values():
        assert isinstance(label, str)
        assert len(label) > 0
        assert " " not in label  # snake_case style label
