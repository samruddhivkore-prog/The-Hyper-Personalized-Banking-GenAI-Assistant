import os
from datetime import date, timedelta

import altair as alt
import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Aisling Bank", page_icon="🏦")

DEMO_CUSTOMERS = ["CUST00001", "CUST00002", "CUST00003", "CUST00098", "CUST00150"]
CATEGORY_COLORS = {
    "groceries": "#4C78A8",
    "transport": "#F58518",
    "dining": "#E45756",
    "bills": "#72B7B2",
    "shopping": "#54A24B",
    "other": "#B279A2",
}


def _api_post(path: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        st.error("We couldn't reach the assistant right now. Please try again in a moment.")
        return None


def _api_get(path: str, params: dict | None = None) -> dict | None:
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        st.error("We couldn't reach the assistant right now. Please try again in a moment.")
        return None


def get_recommendations(customer_id: str) -> dict | None:
    return _api_post("/recommendations", {"customer_id": customer_id})


def ask_chat(customer_id: str, question: str) -> dict | None:
    return _api_post("/chat", {"customer_id": customer_id, "query": question})


def get_spending(customer_id: str, **params) -> dict | None:
    return _api_get(f"/customers/{customer_id}/spending", params=params)


def post_card_action(action: str, customer_id: str, card_id: str) -> dict | None:
    path = "/actions/freeze-card" if action == "freeze_card" else "/actions/unfreeze-card"
    return _api_post(path, {"customer_id": customer_id, "card_id": card_id})


# ---- sidebar: demo customer switcher ----
with st.sidebar:
    st.subheader("Switch demo customer")
    customer_id = st.selectbox("Demo customer", DEMO_CUSTOMERS, label_visibility="collapsed")

if st.session_state.get("_active_customer") != customer_id:
    st.session_state._active_customer = customer_id
    st.session_state.chat_history = []
    st.session_state.recs_response = None
    st.session_state.spending_response = None
    st.session_state.pending_action = None

for key, default in [
    ("chat_history", []),
    ("recs_response", None),
    ("spending_response", None),
    ("pending_action", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---- header row: title left, customer badge right ----
header_left, header_right = st.columns([4, 1])
with header_left:
    st.title("🏦 Aisling Bank Assistant")
with header_right:
    st.markdown(
        f"<div style='text-align:right; padding-top:1.5rem; color:#666;'>"
        f"Customer: <b>{customer_id}</b></div>",
        unsafe_allow_html=True,
    )

# ---- Recommended for you (top, above chat) ----
st.subheader("Recommended for you")
if st.session_state.recs_response is None:
    with st.spinner("Loading recommendations..."):
        st.session_state.recs_response = get_recommendations(customer_id)

recs_result = st.session_state.recs_response
if recs_result and recs_result["recommendations"]:
    cols = st.columns(len(recs_result["recommendations"][:3]))
    for col, rec in zip(cols, recs_result["recommendations"][:3]):
        with col:
            st.markdown(f"**{rec['product']}**")
            st.write(rec["plain_text"])
            with st.expander("Learn more"):
                st.write(rec.get("learn_more", ""))
elif recs_result is not None:
    st.caption("Nothing new to show right now.")

st.divider()

# ---- Your spending ----
st.subheader("Your spending")
month_options = []
today = date.today()
for i in range(12):
    m = today.month - i
    y = today.year
    while m <= 0:
        m += 12
        y -= 1
    month_options.append(f"{y}-{m:02d}")
period_choice = st.selectbox(
    "Period", month_options + ["Custom range"], format_func=lambda v: v if v == "Custom range" else v
)

if period_choice == "Custom range":
    c1, c2 = st.columns(2)
    with c1:
        start_date = st.date_input("Start", value=today - timedelta(days=29))
    with c2:
        end_date = st.date_input("End", value=today)
    spending_params = {"start_date": start_date.isoformat(), "end_date": end_date.isoformat()}
else:
    spending_params = {"month": period_choice}

if st.button("Show spending") or st.session_state.spending_response is None:
    with st.spinner("Loading spending..."):
        st.session_state.spending_response = get_spending(customer_id, **spending_params)

spending = st.session_state.spending_response
if spending:
    st.write(spending["sentence"])
    if spending["has_data"]:
        df = pd.DataFrame(spending["categories"])
        chart = (
            alt.Chart(df)
            .mark_arc()
            .encode(
                theta="amount",
                color=alt.Color(
                    "category",
                    scale=alt.Scale(
                        domain=list(CATEGORY_COLORS.keys()), range=list(CATEGORY_COLORS.values())
                    ),
                ),
                tooltip=["category", "amount", "pct"],
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.table(df.rename(columns={"category": "Category", "amount": "Amount (€)", "pct": "%"}))

st.divider()

# ---- Chat ----
st.subheader("Ask a question")
for role, text, extra in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(text)
        if extra:
            st.caption(extra)

if st.session_state.pending_action:
    pending = st.session_state.pending_action
    st.info(pending["message"])
    yes_col, no_col = st.columns(2)
    if yes_col.button("Yes", key="confirm_yes"):
        result = post_card_action(pending["action"], customer_id, pending["card_id"])
        if result is not None:
            st.session_state.chat_history.append(("assistant", result["message"], None))
            if result["ok"] and pending["action"] == "freeze_card":
                st.session_state.chat_history.append(
                    ("assistant", "Card frozen. Action logged.", None)
                )
        st.session_state.pending_action = None
        st.rerun()
    if no_col.button("No", key="confirm_no"):
        st.session_state.chat_history.append(("assistant", "No problem — no changes made.", None))
        st.session_state.pending_action = None
        st.rerun()

question = st.chat_input("Ask a question, e.g. how do I send money fast?")
if question:
    question = question.replace("$", "€")  # Aisling Bank only deals in EUR
    st.session_state.chat_history.append(("user", question, None))
    with st.spinner("Thinking..."):
        result = ask_chat(customer_id, question)
    if result is not None:
        answer = result["assistant_response"]["answer"]
        sources = result["assistant_response"]["sources"]
        source_caption = f"Source: {', '.join(sources)}" if sources else None
        st.session_state.chat_history.append(("assistant", answer, source_caption))
        if result["assistant_response"].get("pending_action"):
            st.session_state.pending_action = result["assistant_response"]["pending_action"]
        if result["assistant_response"].get("spending_data"):
            st.session_state.spending_response = {
                **result["assistant_response"]["spending_data"],
                "sentence": answer,
                "customer_id": customer_id,
            }
    st.rerun()
