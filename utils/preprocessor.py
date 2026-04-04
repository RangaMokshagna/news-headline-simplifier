# utils/preprocessor.py
# Data loading, cleaning, and validation utilities

import pandas as pd
import re
from io import StringIO

REQUIRED_COLUMNS = {"publish_date", "headline_category", "headline_text"}

# ─── Loader ──────────────────────────────────────────────────────────────────

def load_csv(source) -> tuple:
    """
    Load a CSV from a file path or Streamlit UploadedFile.
    Returns (DataFrame, error_message). On success error_message is None.
    """
    try:
        if isinstance(source, str):
            df = pd.read_csv(source)
        else:
            # Streamlit UploadedFile
            content = source.read().decode("utf-8")
            df = pd.read_csv(StringIO(content))

        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            return None, f"Missing columns: {', '.join(missing)}"

        df = clean_dataframe(df)
        return df, None

    except Exception as e:
        return None, str(e)

# ─── Cleaner ─────────────────────────────────────────────────────────────────

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop nulls, strip whitespace, standardise date and category fields."""
    df = df.dropna(subset=["headline_text", "headline_category"]).copy()

    df["headline_text"] = df["headline_text"].str.strip()
    df["headline_category"] = df["headline_category"].str.strip().str.lower()

    # Parse dates
    if "publish_date" in df.columns:
        df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")

    # Remove empty headlines
    df = df[df["headline_text"].str.len() > 5]
    df = df.reset_index(drop=True)
    return df

# ─── Stats ────────────────────────────────────────────────────────────────────

def dataset_stats(df: pd.DataFrame) -> dict:
    """Compute quick summary statistics for the sidebar/dashboard."""
    if df is None or df.empty:
        return {}

    word_counts = df["headline_text"].apply(lambda t: len(t.split()))

    return {
        "total_headlines": len(df),
        "categories": sorted(df["headline_category"].unique().tolist()),
        "category_counts": df["headline_category"].value_counts().to_dict(),
        "avg_words": round(word_counts.mean(), 1),
        "max_words": int(word_counts.max()),
        "min_words": int(word_counts.min()),
        "date_range": (
            str(df["publish_date"].min().date()) if "publish_date" in df.columns
            and pd.notna(df["publish_date"].min()) else "N/A",
            str(df["publish_date"].max().date()) if "publish_date" in df.columns
            and pd.notna(df["publish_date"].max()) else "N/A",
        ),
    }
