# utils/search.py
# Search Engine: TF-IDF Vectorization + Keyword Matching + Category Filter
# Demonstrates application of search techniques to the headline corpus.

import re
import math
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

STOP_WORDS = set(stopwords.words('english'))

# ─── Text Preprocessing ───────────────────────────────────────────────────────

def preprocess(text: str) -> list:
    """Lowercase, tokenize, remove stopwords and punctuation."""
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t.isalpha() and t not in STOP_WORDS]

# ─── TF-IDF from Scratch ──────────────────────────────────────────────────────

def compute_tf(tokens: list) -> dict:
    """Term Frequency: count of each term / total terms."""
    count = Counter(tokens)
    total = len(tokens)
    return {term: freq / max(total, 1) for term, freq in count.items()}

def compute_idf(corpus_tokens: list) -> dict:
    """Inverse Document Frequency: log(N / df_t) for each term."""
    N = len(corpus_tokens)
    df = {}
    for doc_tokens in corpus_tokens:
        for term in set(doc_tokens):
            df[term] = df.get(term, 0) + 1
    return {term: math.log(N / freq) for term, freq in df.items()}

def compute_tfidf_scores(query: str, corpus: list) -> list:
    """
    For a query and a list of document strings, compute TF-IDF cosine-like
    relevance scores.

    Returns list of (index, score) sorted by score descending.
    """
    query_tokens = preprocess(query)
    corpus_tokens = [preprocess(doc) for doc in corpus]

    idf = compute_idf(corpus_tokens)

    # Query vector
    query_tf = compute_tf(query_tokens)
    query_vec = {t: query_tf[t] * idf.get(t, 0) for t in query_tf}

    scores = []
    for i, doc_tokens in enumerate(corpus_tokens):
        doc_tf = compute_tf(doc_tokens)
        doc_vec = {t: doc_tf[t] * idf.get(t, 0) for t in doc_tf}

        # Dot product
        dot = sum(query_vec.get(t, 0) * doc_vec.get(t, 0) for t in query_vec)

        # Magnitudes
        q_mag = math.sqrt(sum(v ** 2 for v in query_vec.values()))
        d_mag = math.sqrt(sum(v ** 2 for v in doc_vec.values()))

        cosine = dot / (q_mag * d_mag) if q_mag * d_mag > 0 else 0.0
        scores.append((i, round(cosine, 4)))

    return sorted(scores, key=lambda x: x[1], reverse=True)

# ─── Keyword / Boolean Search ─────────────────────────────────────────────────

def keyword_search(query: str, corpus: list, match_all: bool = False) -> list:
    """
    Boolean keyword search.
    match_all=True  → AND logic (all terms must appear)
    match_all=False → OR  logic (any term must appear)
    Returns list of indices where the condition is satisfied.
    """
    query_terms = preprocess(query)
    results = []

    for i, doc in enumerate(corpus):
        doc_terms = set(preprocess(doc))
        if match_all:
            if all(t in doc_terms for t in query_terms):
                results.append(i)
        else:
            if any(t in doc_terms for t in query_terms):
                results.append(i)

    return results

# ─── Combined Search ──────────────────────────────────────────────────────────

def search(
    query: str,
    df: pd.DataFrame,
    category: str = "All",
    method: str = "TF-IDF",
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Main search function used by the Streamlit UI.

    Parameters
    ----------
    query    : user's search string
    df       : DataFrame with columns [publish_date, headline_category, headline_text]
    category : optional category filter ("All" to skip)
    method   : "TF-IDF" | "Keyword (OR)" | "Keyword (AND)"
    top_k    : max results to return

    Returns
    -------
    Filtered and ranked DataFrame.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    # Apply category filter
    filtered = df.copy()
    if category and category != "All":
        filtered = filtered[
            filtered["headline_category"].str.lower() == category.lower()
        ]

    if filtered.empty:
        return filtered

    corpus = filtered["headline_text"].tolist()

    if not query.strip():
        return filtered.head(top_k)

    if method == "TF-IDF":
        scored = compute_tfidf_scores(query, corpus)
        indices = [i for i, s in scored if s > 0][:top_k]
        result = filtered.iloc[indices].copy()
        result["relevance_score"] = [scored[i][1] for i in range(len(indices))]
    else:
        match_all = "AND" in method
        indices = keyword_search(query, corpus, match_all=match_all)[:top_k]
        result = filtered.iloc[indices].copy()
        result["relevance_score"] = 1.0

    return result.reset_index(drop=True)
