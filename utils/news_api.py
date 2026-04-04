"""Utilities for fetching live headlines from NewsAPI."""

from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


def fetch_live_headlines(api_key: str, category: str = "general", count: int = 20):
    """
    Fetch headlines from NewsAPI and return (DataFrame, error_message).

    The returned DataFrame is normalized to:
    - publish_date
    - headline_category
    - headline_text
    """
    try:
        clean_key = (api_key or "").strip()
        if not clean_key:
            return None, "API key is required."

        max_count = max(1, min(int(count), 100))
        params = {
            "apiKey": clean_key,
            "country": "us",
            "category": (category or "general").lower(),
            "pageSize": max_count,
        }
        url = f"https://newsapi.org/v2/top-headlines?{urlencode(params)}"

        with urlopen(url, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))

        if payload.get("status") != "ok":
            return None, payload.get("message", "Failed to fetch headlines from NewsAPI.")

        articles = payload.get("articles", [])
        rows = []
        for article in articles:
            title = (article.get("title") or "").strip()
            if not title or title == "[Removed]":
                continue

            rows.append(
                {
                    "publish_date": article.get("publishedAt"),
                    "headline_category": (category or "general").lower(),
                    "headline_text": title,
                }
            )

        if not rows:
            return None, "No valid headlines returned for this category."

        df = pd.DataFrame(rows)
        df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
        return df, None

    except Exception as e:
        return None, str(e)
