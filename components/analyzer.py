# components/analyzer.py
# NLP Analysis Display: Word, Syntax, Semantic tabs for a single headline

import streamlit as st
import pandas as pd

from nlp.word_analysis import word_level_report
from nlp.syntax_analysis import syntax_report
from nlp.semantic_analysis import semantic_report
from nlp.simplifier import simplify_headline


def render_analyzer(df=None):
    """Render the NLP Analyzer tab."""

    st.markdown("## 🔬 NLP Headline Analyzer")
    st.markdown(
        "Enter any headline below, or pick one from your dataset. "
        "The analyzer breaks it down across three NLP layers and produces a simplified version."
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    input_mode = st.radio("Input Mode", ["Type / Paste", "Pick from Dataset"],
                          horizontal=True)

    headline = ""
    if input_mode == "Type / Paste":
        headline = st.text_area(
            "Enter a headline:",
            placeholder="e.g. Government Officials Deliberate Legislative Amendments...",
            height=80,
        )
    else:
        if df is None or df.empty:
            st.warning("Upload a dataset from the sidebar first.")
            return
        options = df["headline_text"].tolist()
        headline = st.selectbox("Choose a headline:", options)

    if not headline or not headline.strip():
        st.info("👆 Enter or select a headline to begin analysis.")
        return

    # ── Run Pipeline ──────────────────────────────────────────────────────────
    with st.spinner("Running NLP pipeline..."):
        word_data    = word_level_report(headline)
        syntax_data  = syntax_report(headline, word_data["pos_tags"])
        sem_data     = semantic_report(headline, word_data["pos_tags"])
        simp_data    = simplify_headline(headline)

    # ── Simplified Result Banner ──────────────────────────────────────────────
    st.divider()
    st.markdown("### ✅ Simplified Headline")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.success(f"**{simp_data['simplified']}**")
    with col2:
        st.metric("Words Reduced", f"{simp_data['reduction_pct']}%")
        st.metric("Ease (before → after)",
                  f"{simp_data['original_ease']} → {simp_data['simplified_ease']}")

    if simp_data["steps"]:
        with st.expander("🔎 Step-by-Step Simplification Process"):
            for step in simp_data["steps"]:
                st.markdown(f"**{step['name']}**")
                st.markdown(f"- Before: `{step['before']}`")
                st.markdown(f"- After : `{step['after']}`")
                for d in step.get("detail", []):
                    st.caption(f"  ↳ {d}")
                st.markdown("---")

    st.divider()

    # ── Three NLP Tabs ────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "🔤 Word Level Analysis",
        "🌳 Syntax Analysis",
        "🧠 Semantic & Discourse",
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — WORD LEVEL
    # ══════════════════════════════════════════════════════════════════════════
    with tab1:
        st.markdown("#### 1️⃣ Tokenization")
        st.markdown("The headline is split into individual **tokens** (words and punctuation).")
        tokens_display = [f"`{t}`" for t in word_data["tokens"]]
        st.markdown("  ".join(tokens_display))

        st.divider()

        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown("#### 2️⃣ Part-of-Speech Tags")
            pos_rows = [
                {
                    "Word": w,
                    "POS": t,
                    "Meaning": word_data["pos_label_map"].get(t, t),
                }
                for w, t in word_data["pos_tags"]
                if w.isalpha()
            ]
            st.dataframe(pd.DataFrame(pos_rows), use_container_width=True, hide_index=True)

        with col_r:
            st.markdown("#### 3️⃣ Word Frequency")
            freq_df = pd.DataFrame(
                list(word_data["word_frequency"].items()),
                columns=["Word", "Frequency"]
            ).head(10)
            st.dataframe(freq_df, use_container_width=True, hide_index=True)

        st.divider()
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### 4️⃣ Lemmatization")
            st.markdown("Reduces words to their **base form**.")
            st.code(" | ".join(word_data["lemmas"][:12]))

        with col_b:
            st.markdown("#### 5️⃣ Stemming")
            st.markdown("Strips **suffixes** to find the root.")
            st.code(" | ".join(word_data["stems"][:12]))

        st.divider()
        st.markdown("#### 6️⃣ Complex Words Detected")
        if word_data["complex_words"]:
            cw_df = pd.DataFrame(word_data["complex_words"])
            cw_df.columns = ["Complex Word", "Simpler Alternative"]
            st.dataframe(cw_df, use_container_width=True, hide_index=True)
        else:
            st.success("No complex words detected — this headline is already fairly simple!")

        st.markdown(f"**Avg syllables per word:** `{word_data['avg_syllables']}`")

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — SYNTAX
    # ══════════════════════════════════════════════════════════════════════════
    with tab2:
        st.markdown("#### 1️⃣ Sentence Type & Structure")
        struct = syntax_data["structure"]
        scol1, scol2, scol3 = st.columns(3)
        scol1.metric("Sentence Type", struct["sentence_type"])
        scol2.metric("Word Count", struct["word_count"])
        scol3.metric("Modifier Count", struct["modifier_count"])

        st.divider()
        st.markdown("#### 2️⃣ Phrase Chunks")
        chunks = syntax_data["chunks"]
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**🟦 Noun Phrases (NP)**")
            for np in chunks["noun_phrases"] or ["—"]:
                st.markdown(f"- {np}")
        with c2:
            st.markdown("**🟩 Verb Phrases (VP)**")
            for vp in chunks["verb_phrases"] or ["—"]:
                st.markdown(f"- {vp}")
        with c3:
            st.markdown("**🟧 Prepositional Phrases (PP)**")
            for pp in chunks["prep_phrases"] or ["—"]:
                st.markdown(f"- {pp}")

        st.divider()
        st.markdown("#### 3️⃣ Dependency Roles")
        dep_rows = [
            {"Word": w, "POS": t, "Grammatical Role": role}
            for w, t, role in syntax_data["dependencies"]
            if w.isalpha()
        ]
        st.dataframe(pd.DataFrame(dep_rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### 4️⃣ Readability Scores")
        rcol1, rcol2 = st.columns(2)
        rcol1.metric("Flesch Reading Ease", syntax_data["fk_ease"],
                     help="0–100: higher = easier. 60+ is standard English.")
        rcol2.metric("Readability Level", syntax_data["ease_label"])
        st.metric("Flesch-Kincaid Grade Level", syntax_data["fk_grade"],
                  help="Approximate US school grade needed to read this text.")
        st.progress(
            min(int(syntax_data["fk_ease"]), 100),
            text=f"Reading Ease: {syntax_data['fk_ease']}/100"
        )

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — SEMANTIC & DISCOURSE
    # ══════════════════════════════════════════════════════════════════════════
    with tab3:
        st.markdown("#### 1️⃣ Named Entities")
        entities = sem_data["named_entities"]
        if entities:
            ent_df = pd.DataFrame(entities, columns=["Entity", "Type"])
            st.dataframe(ent_df, use_container_width=True, hide_index=True)
        else:
            st.info("No named entities detected.")

        st.divider()
        st.markdown("#### 2️⃣ Topic Detection")
        topics = sem_data["topics"]
        if topics:
            for topic, score in topics:
                st.markdown(f"- **{topic}** — relevance score: `{score}`")
        else:
            st.info("No strong topic match found.")

        st.divider()
        st.markdown("#### 3️⃣ Sentiment Analysis")
        sent = sem_data["sentiment"]
        st.markdown(
            f"{sent['emoji']} **{sent['label']}** "
            f"(score: `{sent['score']}`)"
        )
        if sent["positive_words"]:
            st.markdown(f"Positive signals: {', '.join(f'`{w}`' for w in sent['positive_words'])}")
        if sent["negative_words"]:
            st.markdown(f"Negative signals: {', '.join(f'`{w}`' for w in sent['negative_words'])}")

        st.divider()
        st.markdown("#### 4️⃣ Discourse Markers")
        markers = sem_data["discourse_markers"]
        if markers:
            for category, words in markers.items():
                st.markdown(f"- **{category}**: {', '.join(f'`{w}`' for w in words)}")
        else:
            st.info("No discourse markers found.")

        st.divider()
        st.markdown("#### 5️⃣ Information Density")
        density = sem_data["info_density"]
        dcols = st.columns(4)
        dcols[0].metric("Density Score", f"{density['density_score']}%")
        dcols[1].metric("Nouns", density["nouns"])
        dcols[2].metric("Verbs", density["verbs"])
        dcols[3].metric("Adjectives", density["adjectives"])
        st.markdown(f"**Density Level:** `{density['density_label']}`")
