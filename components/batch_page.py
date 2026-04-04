# components/batch_page.py
# Batch Simplification Page

import streamlit as st
import pandas as pd
from nlp.simplifier import batch_simplify


def render_batch(df):
    """Render the Batch Simplifier tab."""

    st.markdown("## ⚡ Batch Headline Simplifier")
    st.markdown("Simplify multiple headlines at once and download the results as CSV.")

    if df is None or df.empty:
        st.info("📂 Upload a CSV dataset from the sidebar to use batch mode.")
        return

    # ── Controls ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        category = st.selectbox(
            "Filter by Category (optional)",
            ["All"] + sorted(df["headline_category"].unique().tolist()),
            key="batch_cat",
        )
    with col2:
        max_rows = st.slider("Max headlines to simplify", 5, 50, 20)

    filtered = df if category == "All" else df[df["headline_category"] == category]
    subset = filtered.head(max_rows)

    st.markdown(f"Ready to simplify **{len(subset)}** headlines.")

    if st.button("🚀 Run Batch Simplification", type="primary"):
        with st.spinner(f"Simplifying {len(subset)} headlines..."):
            headlines = subset["headline_text"].tolist()
            results = batch_simplify(headlines)

        # ── Build Output DataFrame ─────────────────────────────────────────
        rows = []
        for orig_row, simp in zip(subset.itertuples(index=False), results):
            rows.append({
                "publish_date": getattr(orig_row, "publish_date", ""),
                "category": getattr(orig_row, "headline_category", ""),
                "original_headline": simp["original"],
                "simplified_headline": simp["simplified"],
                "words_before": simp["original_words"],
                "words_after": simp["simplified_words"],
                "reduction_%": simp["reduction_pct"],
                "ease_before": simp["original_ease"],
                "ease_after": simp["simplified_ease"],
            })

        out_df = pd.DataFrame(rows)

        # ── Summary Stats ──────────────────────────────────────────────────
        st.success("✅ Batch simplification complete!")
        m1, m2, m3 = st.columns(3)
        m1.metric("Headlines Processed", len(out_df))
        m2.metric("Avg Word Reduction", f"{out_df['reduction_%'].mean():.1f}%")
        m3.metric("Avg Ease Gain",
                  f"+{(out_df['ease_after'] - out_df['ease_before']).mean():.1f}")

        # ── Preview Table ──────────────────────────────────────────────────
        st.dataframe(out_df, use_container_width=True, hide_index=True)

        # ── Download ──────────────────────────────────────────────────────
        csv = out_df.to_csv(index=False)
        st.download_button(
            "⬇️ Download Results as CSV",
            data=csv,
            file_name="simplified_headlines.csv",
            mime="text/csv",
        )
