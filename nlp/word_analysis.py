# nlp/word_analysis.py
# Word Level Analysis: Tokenization, POS Tagging, Frequency, Stopword Detection

import nltk
import re
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK resources
def download_nltk_resources():
    resources = ['punkt', 'averaged_perceptron_tagger', 'stopwords',
                 'wordnet', 'omw-1.4', 'punkt_tab', 'averaged_perceptron_tagger_eng']
    for r in resources:
        try:
            nltk.download(r, quiet=True)
        except Exception:
            pass

download_nltk_resources()

# ─── Constants ───────────────────────────────────────────────────────────────

# Import the single source-of-truth word map from the simplifier
from nlp.simplifier import WORD_MAP as COMPLEX_WORD_MAP

POS_LABELS = {
    "CC": "Conjunction", "CD": "Number", "DT": "Determiner",
    "EX": "Existential", "FW": "Foreign Word", "IN": "Preposition",
    "JJ": "Adjective", "JJR": "Adj (Comparative)", "JJS": "Adj (Superlative)",
    "LS": "List Item", "MD": "Modal Verb", "NN": "Noun (Singular)",
    "NNS": "Noun (Plural)", "NNP": "Proper Noun", "NNPS": "Proper Noun (Plural)",
    "PDT": "Predeterminer", "POS": "Possessive", "PRP": "Pronoun",
    "PRP$": "Possessive Pronoun", "RB": "Adverb", "RBR": "Adverb (Comparative)",
    "RBS": "Adverb (Superlative)", "RP": "Particle", "SYM": "Symbol",
    "TO": "to", "UH": "Interjection", "VB": "Verb (Base)",
    "VBD": "Verb (Past)", "VBG": "Verb (Gerund)", "VBN": "Verb (Past Participle)",
    "VBP": "Verb (Present)", "VBZ": "Verb (3rd Person)", "WDT": "Wh-Determiner",
    "WP": "Wh-Pronoun", "WP$": "Wh-Possessive", "WRB": "Wh-Adverb",
}

# ─── Core Functions ───────────────────────────────────────────────────────────

def tokenize(text: str) -> list:
    """Split headline into individual word tokens."""
    return word_tokenize(text)

def get_pos_tags(tokens: list) -> list:
    """Assign Part-of-Speech tags to each token."""
    return nltk.pos_tag(tokens)

def get_stopwords() -> set:
    return set(stopwords.words('english'))

def lemmatize_tokens(tokens: list) -> list:
    """Reduce tokens to their base (lemma) form."""
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t.lower()) for t in tokens]

def stem_tokens(tokens: list) -> list:
    """Reduce tokens to their root (stem) form."""
    stemmer = PorterStemmer()
    return [stemmer.stem(t.lower()) for t in tokens]

def word_frequency(tokens: list) -> dict:
    """Count occurrences of each token."""
    clean = [t.lower() for t in tokens if t.isalpha()]
    return dict(Counter(clean).most_common())

def detect_complex_words(tokens: list) -> list:
    """Flag tokens that have simpler alternatives."""
    return [
        {"original": t, "simple": COMPLEX_WORD_MAP[t.lower()]}
        for t in tokens
        if t.lower() in COMPLEX_WORD_MAP
    ]

def syllable_count(word: str) -> int:
    """Approximate syllable count using vowel groups."""
    word = word.lower()
    vowels = "aeiou"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(1, count)

def word_level_report(text: str) -> dict:
    """Full word-level analysis of a headline."""
    tokens = tokenize(text)
    alpha_tokens = [t for t in tokens if t.isalpha()]
    pos_tags = get_pos_tags(tokens)
    stop_words = get_stopwords()
    lemmas = lemmatize_tokens(alpha_tokens)
    stems = stem_tokens(alpha_tokens)
    freq = word_frequency(tokens)
    complex_words = detect_complex_words(tokens)

    stopword_list = [t for t in alpha_tokens if t.lower() in stop_words]
    content_words = [t for t in alpha_tokens if t.lower() not in stop_words]
    avg_syllables = round(
        sum(syllable_count(t) for t in alpha_tokens) / max(len(alpha_tokens), 1), 2
    )

    return {
        "tokens": tokens,
        "alpha_tokens": alpha_tokens,
        "pos_tags": pos_tags,
        "lemmas": lemmas,
        "stems": stems,
        "word_frequency": freq,
        "complex_words": complex_words,
        "stopwords": stopword_list,
        "content_words": content_words,
        "token_count": len(tokens),
        "avg_syllables": avg_syllables,
        "pos_label_map": POS_LABELS,
        "complex_word_map": COMPLEX_WORD_MAP,
    }
