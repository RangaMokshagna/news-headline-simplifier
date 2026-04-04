# nlp/syntax_analysis.py
# Syntax Analysis: Sentence Structure, Phrase Chunking, Readability

import nltk
import re
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# ─── Phrase Chunking Grammar ──────────────────────────────────────────────────

# Noun Phrase: optional determiner + optional adjectives + noun(s)
NP_GRAMMAR = r"""
  NP: {<DT|JJ|JJR|JJS|NNP|NNPS>*<NN|NNS|NNP|NNPS>+}
  VP: {<VB|VBD|VBG|VBN|VBP|VBZ><NP|IN|JJ>*}
  PP: {<IN><NP>}
"""

# ─── Readability Helpers ──────────────────────────────────────────────────────

def _syllable_count(word: str) -> int:
    word = word.lower().strip(".,!?;:")
    vowels = "aeiou"
    count = 0
    prev_v = False
    for ch in word:
        is_v = ch in vowels
        if is_v and not prev_v:
            count += 1
        prev_v = is_v
    return max(1, count)

def _split_sentences(text: str) -> list:
    """Simple regex-based sentence splitter — no NLTK needed."""
    try:
        from nltk.tokenize import sent_tokenize
        return sent_tokenize(text)
    except Exception:
        # Fallback: split on .  !  ?
        parts = re.split(r'[.!?]+', text)
        return [p.strip() for p in parts if p.strip()] or [text]

def _tokenize_words(text: str) -> list:
    """Simple regex-based word tokenizer — no NLTK needed."""
    try:
        from nltk.tokenize import word_tokenize
        return [w for w in word_tokenize(text) if w.isalpha()]
    except Exception:
        return re.findall(r'\b[a-zA-Z]+\b', text)

def flesch_kincaid_grade(text: str) -> float:
    """
    FK Grade Level: measures years of education needed to understand the text.
    Lower = simpler.
    """
    sentences = _split_sentences(text)
    words = _tokenize_words(text)
    syllables = sum(_syllable_count(w) for w in words)

    num_sent = max(len(sentences), 1)
    num_words = max(len(words), 1)

    score = 0.39 * (num_words / num_sent) + 11.8 * (syllables / num_words) - 15.59
    return round(score, 2)

def flesch_reading_ease(text: str) -> float:
    """
    Flesch Reading Ease: 0–100, higher = easier to read.
    60–70 is standard/plain English.
    """
    sentences = _split_sentences(text)
    words = _tokenize_words(text)
    syllables = sum(_syllable_count(w) for w in words)

    num_sent = max(len(sentences), 1)
    num_words = max(len(words), 1)

    score = 206.835 - 1.015 * (num_words / num_sent) - 84.6 * (syllables / num_words)
    return round(max(0, min(100, score)), 2)

def reading_ease_label(score: float) -> str:
    if score >= 90: return "Very Easy"
    if score >= 70: return "Easy"
    if score >= 60: return "Standard"
    if score >= 50: return "Fairly Difficult"
    if score >= 30: return "Difficult"
    return "Very Difficult"

# ─── Syntactic Structure ─────────────────────────────────────────────────────

def get_chunks(pos_tags: list) -> dict:
    """
    Extract Noun Phrases and Verb Phrases using regex chunking.
    Returns labelled chunks with their constituent words.
    """
    parser = nltk.RegexpParser(NP_GRAMMAR)
    tree = parser.parse(pos_tags)

    noun_phrases, verb_phrases, prep_phrases = [], [], []

    for subtree in tree.subtrees():
        if subtree.label() == "NP":
            noun_phrases.append(" ".join(w for w, _ in subtree.leaves()))
        elif subtree.label() == "VP":
            verb_phrases.append(" ".join(w for w, _ in subtree.leaves()))
        elif subtree.label() == "PP":
            prep_phrases.append(" ".join(w for w, _ in subtree.leaves()))

    return {
        "noun_phrases": noun_phrases,
        "verb_phrases": verb_phrases,
        "prep_phrases": prep_phrases,
        "tree": tree,
    }

def sentence_structure(text: str, pos_tags: list) -> dict:
    """
    Classify sentence type and compute structural features.
    """
    words = [w for w, _ in pos_tags if w.isalpha()]
    tags = [t for _, t in pos_tags]

    has_verb = any(t.startswith("VB") for t in tags)
    has_noun = any(t.startswith("NN") for t in tags)
    has_modal = any(t == "MD" for t in tags)
    has_wh = any(t.startswith("W") for t in tags)

    if text.strip().endswith("?") or has_wh:
        sentence_type = "Interrogative"
    elif text.strip().endswith("!"):
        sentence_type = "Exclamatory"
    elif has_verb and not has_noun:
        sentence_type = "Imperative"
    elif has_verb and has_noun:
        sentence_type = "Declarative"
    else:
        sentence_type = "Fragment / Nominal"

    modifiers = [w for w, t in pos_tags if t in ("JJ", "JJR", "JJS", "RB")]

    return {
        "sentence_type": sentence_type,
        "has_verb": has_verb,
        "has_modal": has_modal,
        "modifier_count": len(modifiers),
        "modifiers": modifiers,
        "word_count": len(words),
    }

def dependency_features(pos_tags: list) -> list:
    """
    Approximate dependency roles using POS patterns.
    Returns a list of (word, role) pairs for display.
    """
    result = []
    tags_list = list(pos_tags)
    prev_tag = None

    for i, (word, tag) in enumerate(tags_list):
        if tag in ("NNP", "NNPS"):
            role = "Named Entity / Subject"
        elif tag in ("NN", "NNS") and i == 0:
            role = "Subject"
        elif tag in ("NN", "NNS"):
            role = "Object / Noun"
        elif tag.startswith("VB"):
            role = "Predicate (Verb)"
        elif tag.startswith("JJ"):
            role = "Modifier (Adjective)"
        elif tag.startswith("RB"):
            role = "Modifier (Adverb)"
        elif tag == "IN":
            role = "Relation (Preposition)"
        elif tag == "DT":
            role = "Determiner"
        elif tag == "CD":
            role = "Quantity"
        elif tag == "MD":
            role = "Modal Auxiliary"
        elif tag == "CC":
            role = "Coordinator"
        else:
            role = tag
        result.append((word, tag, role))
        prev_tag = tag

    return result

def syntax_report(text: str, pos_tags: list) -> dict:
    """Full syntax analysis report for one headline."""
    chunks = get_chunks(pos_tags)
    structure = sentence_structure(text, pos_tags)
    deps = dependency_features(pos_tags)
    fk_grade = flesch_kincaid_grade(text)
    fk_ease = flesch_reading_ease(text)

    return {
        "chunks": chunks,
        "structure": structure,
        "dependencies": deps,
        "fk_grade": fk_grade,
        "fk_ease": fk_ease,
        "ease_label": reading_ease_label(fk_ease),
    }
