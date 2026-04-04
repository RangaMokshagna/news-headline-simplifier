# components/search_page.py
# Search Page: TF-IDF and keyword search over the loaded corpus

import streamlit as st
import pandas as pd

from utils.search import search
from nlp.simplifier import simplify_headline


def render_search(df):
    """Render the Search Engine tab."""

    st.markdown("## 🔍 Headline Search Engine")
    st.markdown(
        "Search your corpus using **TF-IDF** (semantic relevance) or "
        "**Keyword** (Boolean) matching — core NLP search techniques."
    )

    if df is None or df.empty:
        st.info("📂 Upload a CSV dataset from the sidebar to enable search.")
        return

    # ── Search Controls ───────────────────────────────────────────────────────
    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        query = st.text_input("🔎 Search query", placeholder="e.g. climate change energy")
    with col2:
        method = st.selectbox("Method", ["TF-IDF", "Keyword (OR)", "Keyword (AND)"])
    with col3:
        category = st.selectbox(
            "Category",
            ["All"] + sorted(df["headline_category"].unique().tolist())
        )

    top_k = st.slider("Max results", 5, 30, 10)

    if not query.strip():
        st.info("Type a query above to search.")
        return

    # ── Run Search ────────────────────────────────────────────────────────────
    with st.spinner("Searching..."):
        results = search(query, df, category=category, method=method, top_k=top_k)

    if results.empty:
        st.warning("No results found. Try different keywords or relax the category filter.")
        return

    st.success(f"Found **{len(results)}** result(s) using **{method}**")

    # ── How It Works ──────────────────────────────────────────────────────────
    with st.expander("ℹ️ How does this search work?"):
        st.markdown("""
**TF-IDF (Term Frequency – Inverse Document Frequency)**
- Measures how important a word is to a specific document relative to the whole corpus.
- Words that appear often in one headline but rarely elsewhere get a **high score**.
- Your query is vectorized and compared to each headline using **cosine similarity**.

**Keyword (OR / AND) Search**
- OR: returns headlines containing **at least one** query word (broad).
- AND: returns headlines containing **all** query words (strict).
- Stopwords (the, a, is…) are ignored.
        """)

    # ── Results Table ─────────────────────────────────────────────────────────
    st.divider()
    for i, row in results.iterrows():
        with st.container():
            cols = st.columns([3, 1])
            with cols[0]:
                st.markdown(f"**{row['headline_text']}**")
                meta = f"📁 `{row['headline_category']}` "
                if pd.notna(row.get("publish_date")):
                    meta += f"| 📅 `{str(row['publish_date'])[:10]}`"
                if "relevance_score" in row:
                    meta += f" | 🎯 Score: `{row['relevance_score']:.4f}`"
                st.caption(meta)

            with cols[1]:
                if st.button("Simplify ✨", key=f"simplify_{i}"):
                    result = simplify_headline(row["headline_text"])
                    st.info(f"**Simplified:** {result['simplified']}")

            st.divider()
