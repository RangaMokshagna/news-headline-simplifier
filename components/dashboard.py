# components/dashboard.py
# Dataset Overview Dashboard page

import streamlit as st
import pandas as pd

def render_dashboard(df: pd.DataFrame, stats: dict):
    """Render the Dataset Overview dashboard tab."""

    st.markdown("## 📊 Dataset Overview")
    st.markdown("Explore the structure and distribution of your news headlines corpus.")

    if df is None or df.empty:
        st.info("📂 Upload a CSV dataset from the sidebar to get started.")
        return

    # ── Summary Metrics ───────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📰 Total Headlines", stats.get("total_headlines", 0))
    col2.metric("📁 Categories", len(stats.get("categories", [])))
    col3.metric("📝 Avg Words", stats.get("avg_words", 0))
    col4.metric("📅 Date Range", f"{stats['date_range'][0]} → {stats['date_range'][1]}")

    st.divider()

    # ── Category Distribution ─────────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 📁 Category Distribution")
        cat_df = pd.DataFrame(
            list(stats.get("category_counts", {}).items()),
            columns=["Category", "Count"]
        ).sort_values("Count", ascending=False)
        st.bar_chart(cat_df.set_index("Category"))

    with col_right:
        st.markdown("### 📋 Category Counts")
        st.dataframe(
            cat_df.reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # ── Headline Length Distribution ──────────────────────────────────────────
    st.markdown("### 📏 Headline Word Count Distribution")
    word_counts = df["headline_text"].apply(lambda t: len(t.split()))
    st.bar_chart(word_counts.value_counts().sort_index())

    st.divider()

    # ── Raw Data Preview ──────────────────────────────────────────────────────
    st.markdown("### 🗂️ Raw Data Preview")

    # Filter controls
    filter_cat = st.selectbox(
        "Filter by Category",
        ["All"] + sorted(df["headline_category"].unique().tolist()),
        key="dashboard_cat_filter",
    )

    preview_df = df if filter_cat == "All" else df[df["headline_category"] == filter_cat]

    st.dataframe(
        preview_df[["publish_date", "headline_category", "headline_text"]].head(20),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Showing up to 20 of {len(preview_df)} headlines")
