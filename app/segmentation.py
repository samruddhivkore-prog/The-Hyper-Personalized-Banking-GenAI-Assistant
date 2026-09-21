"""Customer segmentation: KMeans over encoded profile features.

Deterministic, classical ML — cheap, fast, and auditable. Runs offline on a
schedule (see scripts/offline_refresh.py), not per-request.
"""
from dataclasses import dataclass

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "age",
    "income_bracket_ordinal",
    "account_tenure_months",
    "avg_monthly_transactions",
    "num_products_held",
    "channel_digital",
]

INCOME_ORDER = ["under_25k", "25k_50k", "50k_75k", "75k_100k", "100k_plus"]


def _income_ordinal(bracket: str) -> int:
    try:
        return INCOME_ORDER.index(bracket)
    except ValueError:
        return len(INCOME_ORDER) // 2


def encode_features(profiles: list[dict]) -> pd.DataFrame:
    rows = []
    for p in profiles:
        rows.append(
            {
                "customer_id": p["customer_id"],
                "age": p["age"],
                "income_bracket_ordinal": _income_ordinal(p["income_bracket"]),
                "account_tenure_months": p["account_tenure_months"],
                "avg_monthly_transactions": p["avg_monthly_transactions"],
                "num_products_held": len(p.get("products_held") or []),
                "channel_digital": 1 if p.get("channel_preference") in ("mobile", "web") else 0,
            }
        )
    return pd.DataFrame(rows).set_index("customer_id")


def _label_segment(centroid: dict) -> str:
    """Turn a cluster centroid (in original feature units) into a human label."""
    age = centroid["age"]
    digital = centroid["channel_digital"]
    tenure = centroid["account_tenure_months"]
    txn = centroid["avg_monthly_transactions"]
    income = centroid["income_bracket_ordinal"]

    age_part = "young" if age < 35 else ("mid_career" if age < 55 else "senior")
    channel_part = "digital_first" if digital >= 0.5 else "branch_oriented"

    if txn >= 30:
        activity_part = "high_activity"
    elif txn >= 12:
        activity_part = "regular"
    else:
        activity_part = "low_activity"

    if income >= 3:
        wealth_part = "affluent"
    elif tenure >= 60:
        wealth_part = "loyal"
    else:
        wealth_part = None

    parts = [channel_part, age_part]
    if wealth_part:
        parts.append(wealth_part)
    parts.append(activity_part) if activity_part != "regular" else None
    return "_".join(parts)


@dataclass
class SegmentationResult:
    assignments: dict[str, int]
    labels: dict[int, str]
    silhouette: float | None


class SegmentationModel:
    def __init__(self, k: int = 5, random_state: int = 42):
        self.k = k
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model: KMeans | None = None

    def fit(self, profiles: list[dict]) -> SegmentationResult:
        features = encode_features(profiles)
        if len(features) < self.k:
            # not enough customers to form k clusters; fall back to 1 cluster per customer group
            effective_k = max(1, len(features))
        else:
            effective_k = self.k

        X = self.scaler.fit_transform(features[FEATURE_COLUMNS])
        self.model = KMeans(n_clusters=effective_k, random_state=self.random_state, n_init=10)
        cluster_ids = self.model.fit_predict(X)

        silhouette = None
        if effective_k > 1 and len(features) > effective_k:
            silhouette = float(silhouette_score(X, cluster_ids))

        centroids_original = self.scaler.inverse_transform(self.model.cluster_centers_)
        labels = {}
        for cid in range(effective_k):
            centroid = dict(zip(FEATURE_COLUMNS, centroids_original[cid]))
            labels[cid] = _label_segment(centroid)

        assignments = {
            cust_id: int(cid) for cust_id, cid in zip(features.index, cluster_ids)
        }
        return SegmentationResult(assignments=assignments, labels=labels, silhouette=silhouette)

    def predict_one(self, profile: dict, fitted: SegmentationResult) -> tuple[int, str]:
        """Cheap path used when we already have a fitted result for a batch predict."""
        cid = fitted.assignments[profile["customer_id"]]
        return cid, fitted.labels[cid]
