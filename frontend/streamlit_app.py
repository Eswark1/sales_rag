"""
Streamlit app – two tabs:
  1. Chatbot  (executive NL questions)
  2. Dashboard (pre-built KPI charts)
"""

import os

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Defaults to Docker-internal name; override with API_URL env var for local dev:
#   export API_URL=http://localhost:8000
API_URL = os.environ.get("API_URL", "http://app:8000")

st.set_page_config(page_title="Sales Intelligence", layout="wide", page_icon="📊")
st.title("Sales Intelligence Platform")

tab_chat, tab_dash = st.tabs(["💬 Ask a Question", "📊 Executive Dashboard"])

# ─────────────────────────────────────────────────────────────
# TAB 1 – Chatbot
# ─────────────────────────────────────────────────────────────
with tab_chat:
    st.subheader("Ask the sales database anything")
    st.caption("Powered by DeepSeek · TAG (Text-to-SQL) + RAG (context retrieval)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sql"):
                with st.expander("View SQL", expanded=False):
                    st.code(msg["sql"], language="sql")
            if msg.get("rows"):
                with st.expander("View raw data", expanded=False):
                    st.dataframe(pd.DataFrame(msg["rows"]), use_container_width=True)
            if msg.get("context"):
                with st.expander("View retrieved context snippets", expanded=False):
                    for snip in msg["context"]:
                        st.text(snip)

    if prompt := st.chat_input("e.g. What is total revenue by region this year?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/ask",
                        json={"question": prompt, "use_rag": True},
                        timeout=120,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    st.markdown(data["answer"])
                    if data.get("sql"):
                        with st.expander("View SQL", expanded=False):
                            st.code(data["sql"], language="sql")
                    if data.get("rows"):
                        with st.expander("View raw data", expanded=False):
                            st.dataframe(pd.DataFrame(data["rows"]), use_container_width=True)
                    if data.get("context"):
                        with st.expander("View context snippets", expanded=False):
                            for snip in data["context"]:
                                st.text(snip)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": data["answer"],
                        "sql": data.get("sql"),
                        "rows": data.get("rows"),
                        "context": data.get("context"),
                    })
                except Exception as e:
                    st.error(f"API error: {e}")

# ─────────────────────────────────────────────────────────────
# TAB 2 – Dashboard
# ─────────────────────────────────────────────────────────────
with tab_dash:
    st.subheader("Executive KPI Dashboard")

    def _query(sql: str) -> pd.DataFrame:
        try:
            r = requests.post(
                f"{API_URL}/ask",
                json={"question": sql, "use_rag": False},
                timeout=60,
            )
            r.raise_for_status()
            rows = r.json().get("rows", [])
            return pd.DataFrame(rows) if rows else pd.DataFrame()
        except Exception:
            return pd.DataFrame()

    # Use direct SQL questions that the TAG agent can handle exactly
    PIPELINE_Q  = "Show total estimated_value and count of opportunities grouped by stage"
    REVENUE_Q   = "Show total total_amount as revenue grouped by region name"
    WINRATE_Q   = "Show count of opportunities for Closed Won and Closed Lost stages"
    REP_Q       = "Show total total_amount as revenue grouped by rep full_name order by revenue desc limit 10"
    MONTHLY_Q   = "Show total total_amount as revenue grouped by strftime year-month of order_date order by month"

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Pipeline by Stage")
        df_pipe = _query(PIPELINE_Q)
        if not df_pipe.empty:
            fig = px.funnel(df_pipe, x=df_pipe.columns[1], y=df_pipe.columns[0],
                            color_discrete_sequence=px.colors.sequential.Blues_r)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No pipeline data")

    with col2:
        st.markdown("#### Revenue by Region")
        df_rev = _query(REVENUE_Q)
        if not df_rev.empty:
            fig = px.bar(df_rev, x=df_rev.columns[0], y=df_rev.columns[1],
                         color=df_rev.columns[0],
                         color_discrete_sequence=px.colors.qualitative.Set2)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No revenue data")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("#### Win Rate")
        df_wr = _query(WINRATE_Q)
        if not df_wr.empty:
            fig = px.pie(df_wr, names=df_wr.columns[0], values=df_wr.columns[1],
                         hole=0.4, color_discrete_map={
                             "Closed Won": "#2ecc71", "Closed Lost": "#e74c3c"
                         })
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No win/loss data")

    with col4:
        st.markdown("#### Top Reps by Revenue")
        df_rep = _query(REP_Q)
        if not df_rep.empty:
            fig = px.bar(df_rep, x=df_rep.columns[1], y=df_rep.columns[0],
                         orientation="h",
                         color_discrete_sequence=["#3498db"])
            fig.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No rep data")

    st.markdown("#### Monthly Revenue Trend")
    df_monthly = _query(MONTHLY_Q)
    if not df_monthly.empty:
        fig = px.line(df_monthly, x=df_monthly.columns[0], y=df_monthly.columns[1],
                      markers=True, color_discrete_sequence=["#9b59b6"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No monthly data")
