"""Staff-only admin console — a separate Streamlit app (run on its own port,
e.g. 8502) so nothing here is reachable from the customer-facing app in
frontend/streamlit_app.py. Password-gated via ADMIN_PASSWORD if set."""
import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

st.set_page_config(page_title="Aisling Bank — Admin", page_icon="🛠️")


def _get(path: str, params: dict | None = None):
    try:
        resp = requests.get(f"{API_BASE_URL}{path}", params=params, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


def _post(path: str, payload: dict):
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


if ADMIN_PASSWORD:
    if "admin_authed" not in st.session_state:
        st.session_state.admin_authed = False
    if not st.session_state.admin_authed:
        st.title("🛠️ Aisling Bank — Admin")
        pw = st.text_input("Password", type="password")
        if st.button("Log in"):
            if pw == ADMIN_PASSWORD:
                st.session_state.admin_authed = True
                st.rerun()
            else:
                st.error("Incorrect password.")
        st.stop()
else:
    st.warning("ADMIN_PASSWORD is not set — this admin app is open with no password.")

st.title("🛠️ Aisling Bank — Admin")

status = _get("/admin/status")
if status:
    st.caption(f"Active LLM: **{status['llm_provider']}**")

tab_segments, tab_customer, tab_fairness, tab_actions = st.tabs(
    ["Segments", "Customer lookup", "Fairness report", "Card actions log"]
)

with tab_segments:
    st.subheader("Customer segments")
    segments = _get("/segments")
    if segments:
        st.table(pd.DataFrame(segments))

with tab_customer:
    st.subheader("Per-customer recommendations (raw)")
    customer_id = st.text_input("Customer ID", value="CUST00001")
    if st.button("Look up"):
        result = _post("/recommendations", {"customer_id": customer_id})
        if result:
            st.markdown(f"**Segment:** {result.get('segment') or 'unassigned'}")
            for rec in result["recommendations"]:
                st.markdown(
                    f"- **{rec['product']}** — score `{rec['score']}`, "
                    f"reason_code `{rec['reason_code']}`"
                )
            st.markdown("**Fairness flags:**")
            if result["fairness_flags"]:
                for flag in result["fairness_flags"]:
                    st.warning(flag)
            else:
                st.caption("None raised.")

with tab_fairness:
    st.subheader("Latest fairness report")
    reports = _get("/admin/fairness")
    if reports:
        for report in reports:
            st.markdown(f"**Dimension:** {report['group_by']} (threshold {report['threshold']})")
            st.table(pd.DataFrame(report["products"]))
    else:
        st.caption("No fairness report available yet — run scripts/offline_refresh.py.")

with tab_actions:
    st.subheader("Card actions log")
    log = _get("/admin/actions-log")
    if log:
        st.table(pd.DataFrame(log))
    else:
        st.caption("No actions logged yet.")
