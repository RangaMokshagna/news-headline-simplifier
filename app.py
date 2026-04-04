# app.py
# News Headline Simplifier — Main Streamlit Application
# Run: streamlit run app.py

import streamlit as st
import os

from utils.preprocessor import load_csv, dataset_stats
from components.dashboard import render_dashboard
from components.analyzer import render_analyzer
from components.search_page import render_search
from components.batch_page import render_batch

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News Headline Simplifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* App background */
.stApp { background-color: #0f1117; }

/* Main header */
.main-header {
    background: linear-gradient(135deg, #1a1f2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #1e3a5f;
}
.main-header h1 {
    color: #e8f4fd;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}
.main-header p {
    color: #8ab4d4;
    font-size: 1rem;
    margin: 0.5rem 0 0 0;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #1f2937;
}

/* Metric cards */
[data-testid="metric-container"] {
    background-color: #1a2332;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 0.8rem;
}

/* Tab labels */
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.9rem;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 8px;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📰 News Headline Simplifier</h1>
    <p>
        NLP-powered tool · Word Level · Syntax · Semantic & Discourse Analysis ·
        TF-IDF Search · Batch Processing
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df = None
if "stats" not in st.session_state:
    st.session_state.stats = {}

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Data Source")

    data_source = st.radio("Load dataset from:", ["Sample Dataset", "Upload CSV", "🔴 Live News API"])

    if data_source == "Sample Dataset":
        sample_path = os.path.join(os.path.dirname(__file__), "data", "sample_dataset.csv")
        df, err = load_csv(sample_path)
        if err:
            st.error(f"Error loading sample: {err}")
        else:
            st.session_state.df = df
            st.session_state.stats = dataset_stats(df)
            st.success(f"✅ Loaded {len(df)} headlines")

    elif data_source == "Upload CSV":
        uploaded = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            help="Required columns: publish_date, headline_category, headline_text",
        )
        if uploaded:
            df, err = load_csv(uploaded)
            if err:
                st.error(f"❌ {err}")
            else:
                st.session_state.df = df
                st.session_state.stats = dataset_stats(df)
                st.success(f"✅ Loaded {len(df)} headlines")

    elif data_source == "🔴 Live News API":
        st.markdown("#### 🌐 NewsAPI Integration")
        default_api_key = os.getenv("NEWS_API_KEY", "")
        if not default_api_key:
            try:
                default_api_key = st.secrets.get("NEWS_API_KEY", "")
            except Exception:
                default_api_key = ""
        api_key = st.text_input(
            "Paste your API Key",
            type="password",
            placeholder="Get free key at newsapi.org",
            value=default_api_key,
        )
        news_category = st.selectbox(
            "Category",
            ["general", "technology", "health", "business", "sports", "science"],
        )
        count = st.slider("Number of headlines", 5, 50, 20)

        if api_key:
            if st.button("🔴 Fetch Live Headlines", type="primary"):
                from utils.news_api import fetch_live_headlines
                with st.spinner("Fetching live news..."):
                    live_df, err = fetch_live_headlines(api_key, news_category, count)
                if err:
                    st.error(f"❌ {err}")
                else:
                    st.session_state.df = live_df
                    st.session_state.stats = dataset_stats(live_df)
                    st.success(f"✅ Loaded {len(live_df)} live headlines!")
        else:
            st.info("🔑 Enter your API key above\n\nGet free key → [newsapi.org](https://newsapi.org)")

    st.divider()

    # ── Dataset Info ──────────────────────────────────────────────────────────
    if st.session_state.df is not None:
        stats = st.session_state.stats
        st.markdown("### 📊 Dataset Info")
        st.markdown(f"**Headlines:** {stats.get('total_headlines', 0)}")
        st.markdown(f"**Categories:** {len(stats.get('categories', []))}")
        st.markdown(f"**Avg words:** {stats.get('avg_words', 0)}")
        st.markdown(f"**Date range:** {stats['date_range'][0]} → {stats['date_range'][1]}")

    st.divider()

    # ── NLP Pipeline Explanation ───────────────────────────────────────────────
    with st.expander("❓ How the NLP Pipeline Works"):
        st.markdown("""
**1. Word Level Analysis**
- Tokenization → splits headline into words
- POS Tagging → labels each word (Noun, Verb, Adj…)
- Lemmatization & Stemming → reduce to base form
- Complex word detection → finds hard words

**2. Syntax Analysis**
- Phrase chunking (NP, VP, PP)
- Sentence type classification
- Dependency role assignment
- Readability scoring (Flesch-Kincaid)

**3. Semantic & Discourse**
- Named Entity Recognition (NER)
- Topic detection (keyword scoring)
- Sentiment analysis (lexicon-based)
- Discourse marker detection
- Information density measurement

**4. Simplification Pipeline**
1. Replace verbose phrases
2. Swap complex words with simpler ones
3. Remove redundant modifiers
4. Clean & reconstruct

**5. Search Techniques**
- TF-IDF cosine similarity (semantic rank)
- Keyword Boolean OR/AND search
        """)

    st.divider()
    st.caption("Built with NLTK · Streamlit · Python")

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Dataset Overview",
    "🔬 NLP Analyzer",
    "🔍 Search Engine",
    "⚡ Batch Simplifier",
])

with tab1:
    render_dashboard(st.session_state.df, st.session_state.stats)

with tab2:
    render_analyzer(st.session_state.df)

with tab3:
    render_search(st.session_state.df)

with tab4:
    render_batch(st.session_state.df)
