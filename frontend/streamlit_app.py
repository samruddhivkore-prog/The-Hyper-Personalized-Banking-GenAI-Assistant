import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="Hyper-Personalized Banking Assistant", page_icon="🏦")
st.title("🏦 Hyper-Personalized Banking Assistant")
st.caption("RAG-grounded, fairness-audited recommendations — demo UI")

with st.sidebar:
    st.header("Customer")
    customer_id = st.text_input("Customer ID", value="CUST00001")
    st.caption(f"API: {API_BASE_URL}")
    if st.button("Check API health"):
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
            st.json(resp.json())
        except requests.RequestException as e:
            st.error(f"API unreachable: {e}")

if "history" not in st.session_state:
    st.session_state.history = []


def call_recommendations(cid: str, query: str | None):
    payload = {"customer_id": cid}
    if query:
        payload["query"] = query
    resp = requests.post(f"{API_BASE_URL}/recommendations", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


st.subheader("Personalized recommendations")
if st.button("Get recommendations (no question)"):
    try:
        result = call_recommendations(customer_id, None)
        st.session_state.history.append(("summary", result))
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")

st.subheader("Ask a question")
query = st.text_input("Question", value="What's the fastest way to send $2,000 to my sister's account?")
if st.button("Ask"):
    try:
        result = call_recommendations(customer_id, query)
        st.session_state.history.append(("chat", result))
    except requests.RequestException as e:
        st.error(f"Request failed: {e}")

for kind, result in reversed(st.session_state.history):
    st.divider()
    segment = result.get("segment") or "unassigned"
    st.markdown(f"**Customer:** {result['customer_id']}  |  **Segment:** {segment}")

    st.markdown("**Recommendations:**")
    for rec in result["recommendations"]:
        st.markdown(f"- {rec['product']} (score={rec['score']}, reason: `{rec['reason_code']}`)")

    st.markdown("**Assistant response:**")
    st.info(result["assistant_response"]["answer"])
    if result["assistant_response"]["sources"]:
        st.caption("Sources: " + ", ".join(result["assistant_response"]["sources"]))
    st.caption(f"Confidence: {result['assistant_response']['confidence']}")

    if result["fairness_flags"]:
        st.warning("Fairness flags:\n" + "\n".join(f"- {f}" for f in result["fairness_flags"]))
