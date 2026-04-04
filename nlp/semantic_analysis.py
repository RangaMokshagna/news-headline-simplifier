# nlp/semantic_analysis.py
# Semantic & Discourse Analysis: NER, Topic Detection, Sentiment, Coherence

import re
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ─── Named Entity Recognition (Rule-Based) ────────────────────────────────────

# Patterns for common headline entity types
ENTITY_PATTERNS = {
    "ORG": [
        r'\b(Government|Parliament|Senate|Congress|Commission|Federation|Organization|'
        r'Authority|Agency|Institute|Corporation|Committee|Council|Ministry|Department|'
        r'Bureau|Foundation|Association|Union|Alliance)\b'
    ],
    "PERSON_ROLE": [
        r'\b(Officials?|Researchers?|Scientists?|Professionals?|Experts?|Authorities?|'
        r'Ministers?|Presidents?|Doctors?|Analysts?|Investigators?|Representatives?)\b'
    ],
    "TOPIC_DOMAIN": [
        r'\b(AI|Artificial Intelligence|Machine Learning|Cybersecurity|Blockchain|'
        r'Quantum Computing|Climate Change|COVID|Pandemic|Economy|Election)\b'
    ],
    "GEO": [
        r'\b(Global|International|National|Federal|Municipal|Regional|Continental|'
        r'Metropolitan|Urban|Rural)\b'
    ],
}

def extract_named_entities(text: str) -> list:
    """
    Rule-based NER: find organizations, roles, geographic terms, topic domains.
    Returns list of (entity_text, entity_type) pairs.
    """
    found = []
    for entity_type, patterns in ENTITY_PATTERNS.items():
        for pattern in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                found.append((match.group(), entity_type))
    # Remove duplicates preserving order
    seen = set()
    unique = []
    for item in found:
        key = item[0].lower()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique

# ─── Topic Detection ─────────────────────────────────────────────────────────

TOPIC_KEYWORDS = {
    "Politics & Government": [
        "government", "parliament", "election", "policy", "legislation", "senate",
        "congress", "minister", "political", "vote", "democratic", "regulatory",
        "administration", "judicial", "constitutional", "reform", "diplomatic"
    ],
    "Technology & AI": [
        "ai", "artificial intelligence", "technology", "digital", "cyber", "software",
        "algorithm", "machine learning", "computing", "data", "internet", "robot",
        "automation", "blockchain", "quantum", "semiconductor", "cloud"
    ],
    "Health & Medicine": [
        "health", "medical", "hospital", "disease", "vaccine", "drug", "cancer",
        "pharmaceutical", "clinical", "patient", "treatment", "mental", "pandemic",
        "epidemic", "diagnostic", "therapeutic", "neurological", "cardiovascular"
    ],
    "Business & Economy": [
        "business", "economy", "market", "investment", "financial", "trade", "bank",
        "company", "corporate", "inflation", "stock", "merger", "acquisition",
        "revenue", "profit", "startup", "venture", "economic"
    ],
    "Environment & Climate": [
        "climate", "environment", "carbon", "emission", "renewable", "energy",
        "ocean", "forest", "biodiversity", "pollution", "glacier", "coral",
        "sustainability", "conservation", "ecosystem", "fossil fuel"
    ],
    "Sports": [
        "sports", "athlete", "championship", "olympic", "football", "basketball",
        "tournament", "competition", "team", "game", "match", "player", "coach",
        "world cup", "league", "athletic"
    ],
}

def detect_topics(text: str) -> list:
    """Score each topic by keyword overlap with the headline."""
    text_lower = text.lower()
    scores = {}
    for topic, keywords in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        if score > 0:
            scores[topic] = score
    # Sort by score descending
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ─── Sentiment Analysis (Lexicon-Based) ──────────────────────────────────────

POSITIVE_WORDS = {
    "achieve", "advance", "benefit", "boost", "celebrate", "confirm", "create",
    "develop", "enhance", "expand", "gain", "grow", "help", "improve", "increase",
    "innovate", "launch", "lead", "positive", "progress", "protect", "record",
    "reduce", "reform", "resolve", "restore", "save", "secure", "strengthen",
    "succeed", "support", "sustainable", "thrive", "win",
}

NEGATIVE_WORDS = {
    "attack", "ban", "breach", "collapse", "concern", "conflict", "controversy",
    "crisis", "danger", "decline", "deficit", "delay", "devastate", "disaster",
    "dispute", "drop", "fail", "fall", "fraud", "halt", "harm", "investigate",
    "kill", "lack", "lose", "problem", "protest", "restrict", "risk", "scandal",
    "shortage", "struggle", "threat", "violate", "warn", "worsen",
}

def analyze_sentiment(text: str) -> dict:
    """Lexicon-based polarity scoring for a headline."""
    tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    pos_hits = [t for t in tokens if t in POSITIVE_WORDS]
    neg_hits = [t for t in tokens if t in NEGATIVE_WORDS]

    score = len(pos_hits) - len(neg_hits)
    if score > 0:
        label, emoji = "Positive", "🟢"
    elif score < 0:
        label, emoji = "Negative", "🔴"
    else:
        label, emoji = "Neutral", "🟡"

    return {
        "label": label,
        "emoji": emoji,
        "score": score,
        "positive_words": pos_hits,
        "negative_words": neg_hits,
    }

# ─── Discourse / Coherence Features ──────────────────────────────────────────

DISCOURSE_MARKERS = {
    "Cause/Effect": ["because", "therefore", "thus", "hence", "as a result", "due to",
                     "leads to", "causes", "results in", "attributable"],
    "Contrast": ["but", "however", "although", "despite", "yet", "while", "whereas",
                 "nevertheless", "on the other hand", "amid"],
    "Addition": ["and", "also", "furthermore", "moreover", "additionally", "as well"],
    "Condition": ["if", "unless", "provided", "given", "following", "after", "amid"],
    "Temporal": ["after", "before", "during", "while", "when", "as", "following",
                 "upcoming", "recent", "latest"],
}

def detect_discourse_markers(text: str) -> dict:
    """Identify discourse/coherence markers in the headline."""
    text_lower = text.lower()
    found = {}
    for category, markers in DISCOURSE_MARKERS.items():
        hits = [m for m in markers if re.search(r'\b' + re.escape(m) + r'\b', text_lower)]
        if hits:
            found[category] = hits
    return found

def information_density(text: str, pos_tags: list) -> dict:
    """
    Measure how information-dense a headline is.
    High noun/adjective ratio → complex, information-packed.
    """
    total = len([w for w, _ in pos_tags if w.isalpha()])
    nouns = len([w for w, t in pos_tags if t.startswith("NN")])
    verbs = len([w for w, t in pos_tags if t.startswith("VB")])
    adjs = len([w for w, t in pos_tags if t.startswith("JJ")])
    advs = len([w for w, t in pos_tags if t.startswith("RB")])
    preps = len([w for w, t in pos_tags if t == "IN"])

    density_score = round((nouns + adjs) / max(total, 1) * 100, 1)

    return {
        "density_score": density_score,
        "nouns": nouns,
        "verbs": verbs,
        "adjectives": adjs,
        "adverbs": advs,
        "prepositions": preps,
        "total_content_words": total,
        "density_label": "High" if density_score > 50 else "Medium" if density_score > 30 else "Low",
    }

def semantic_report(text: str, pos_tags: list) -> dict:
    """Full semantic and discourse analysis report."""
    return {
        "named_entities": extract_named_entities(text),
        "topics": detect_topics(text),
        "sentiment": analyze_sentiment(text),
        "discourse_markers": detect_discourse_markers(text),
        "info_density": information_density(text, pos_tags),
    }
