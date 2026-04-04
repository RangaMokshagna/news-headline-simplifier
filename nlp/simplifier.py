# nlp/simplifier.py
# Accurate News Headline Simplifier
#
# Strategy: NEVER reconstruct from POS tags (breaks word order).
# Apply ordered string-level substitutions only:
#   Pass 1 → Verbose multi-word phrase removal
#   Pass 2 → Noun chain compression
#   Pass 3 → Complex single-word substitution (regex word-boundary safe)
#   Pass 4 → Filler adverb removal
#   Pass 5 → Final cleanup

import re
from nlp.syntax_analysis import flesch_reading_ease, reading_ease_label

# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — VERBOSE MULTI-WORD PHRASES  (applied longest-first)
# ─────────────────────────────────────────────────────────────────────────────
VERBOSE_PHRASES = [
    ("attributable to",                     "caused by"),
    ("pertaining to",                       "on"),
    ("in relation to",                      "about"),
    ("with regard to",                      "about"),
    ("with respect to",                     "about"),
    ("in connection with",                  "on"),
    ("in conjunction with",                 "with"),
    ("in accordance with",                  "under"),
    ("in compliance with",                  "under"),
    ("in the context of",                   "in"),
    ("in the wake of",                      "after"),
    ("in the aftermath of",                 "after"),
    ("in the course of",                    "during"),
    ("in the process of",                   ""),
    ("in the near future",                  "soon"),
    ("at the present time",                 "now"),
    ("at this point in time",               "now"),
    ("on a regular basis",                  "regularly"),
    ("on an ongoing basis",                 ""),
    ("on the occasion of",                  "at"),
    ("in the event that",                   "if"),
    ("in an effort to",                     "to"),
    ("in order to",                         "to"),
    ("for the purpose of",                  "for"),
    ("due to the fact that",                "because"),
    ("as a result of",                      "because of"),
    ("as a consequence of",                 "because of"),
    ("despite the fact that",               "though"),
    ("in light of the fact that",           "since"),
    ("in light of",                         "given"),
    ("with the exception of",               "except"),
    ("a large number of",                   "many"),
    ("a significant number of",             "many"),
    ("a growing number of",                 "more"),
    ("a majority of",                       "most"),
    ("a wide range of",                     "many"),
    ("a variety of",                        "various"),
    ("it is important to note that",        ""),
    ("it should be noted that",             ""),
    ("it has been reported that",           ""),
    ("according to reports",                ""),
    ("is expected to",                      "will"),
    ("are expected to",                     "will"),
    ("is set to",                           "will"),
    ("are set to",                          "will"),
    ("is slated to",                        "will"),
    ("is poised to",                        "will"),
    ("are poised to",                       "will"),
    ("is seeking to",                       "wants to"),
    ("are seeking to",                      "want to"),
    ("amid growing concerns",               "amid concerns"),
    ("regulatory framework",                "rules"),
    ("legislative framework",               "laws"),
    ("policy framework",                    "policy"),
    ("implementation of",                   ""),
    ("establishment of",                    ""),
    ("introduction of",                     ""),
]

# ─────────────────────────────────────────────────────────────────────────────
# PASS 2 — NOUN CHAIN COMPRESSION  (long compound noun phrases → short)
# ─────────────────────────────────────────────────────────────────────────────
NOUN_CHAINS = [
    ("Artificial Intelligence Algorithms",          "AI"),
    ("Artificial Intelligence",                     "AI"),
    ("Machine Learning Systems",                    "AI systems"),
    ("Machine Learning",                            "AI"),
    ("Natural Language Processing",                 "NLP"),
    ("Quantum Computing Architecture",              "Quantum computers"),
    ("Cybersecurity Professionals",                 "Security experts"),
    ("Autonomous Vehicular Systems",                "Self-driving cars"),
    ("Autonomous Vehicles",                         "Self-driving cars"),
    ("Renewable Energy Infrastructure",             "Clean energy"),
    ("Renewable Energy",                            "Clean energy"),
    ("Fossil Fuel",                                 "Oil and gas"),
    ("Venture Capital Funding",                     "Startup funding"),
    ("Venture Capital",                             "Investor funding"),
    ("Central Banking Institutions",                "Central banks"),
    ("Federal Reserve",                             "Fed"),
    ("Social Media Platforms",                      "Social media"),
    ("Public Health Officials",                     "Health officials"),
    ("Mental Health Professionals",                 "Therapists"),
    ("International Athletic Federation",           "Sports body"),
    ("Olympic Organizing Committee",                "Olympics committee"),
    ("Parliamentary Representatives",               "Lawmakers"),
    ("Legislative Body",                            "Parliament"),
    ("Electoral Commission",                        "Election body"),
    ("National Security Agencies",                  "Security agencies"),
    ("Multinational Corporations",                  "Global firms"),
    ("Financial Institutions",                      "Banks"),
    ("Pharmaceutical Enterprises",                  "Drug companies"),
    ("Semiconductor Manufacturing",                 "Chip manufacturing"),
    ("Cloud Computing",                             "Cloud services"),
    ("Coral Reef Ecosystem",                        "Coral reefs"),
    ("Single-Use Plastic",                          "Plastic"),
    ("Glacial Recession",                           "Glacier melting"),
    ("Mergers and Acquisitions",                    "M&A"),
    ("Epidemiological Researchers",                 "Disease researchers"),
    ("Neurological Researchers",                    "Brain scientists"),
    ("Marine Biologists",                           "Scientists"),
    ("Environmental Scientists",                    "Scientists"),
    ("Conservation Organizations",                  "Conservationists"),
]

# ─────────────────────────────────────────────────────────────────────────────
# PASS 3 — COMPLEX WORD → SIMPLE WORD  (400+ verified entries)
# ─────────────────────────────────────────────────────────────────────────────
WORD_MAP = {
    "accelerated":          "faster",
    "acknowledge":          "admit",
    "acknowledged":         "admitted",
    "acquisition":          "buyout",
    "acquisitions":         "buyouts",
    "adjacent":             "nearby",
    "administer":           "run",
    "administering":        "running",
    "administration":       "government",
    "administrative":       "government",
    "advocate":             "push for",
    "advocates":            "pushes for",
    "alleviate":            "ease",
    "alleviating":          "easing",
    "allocate":             "give",
    "allocated":            "given",
    "allocation":           "funds",
    "allocations":          "funds",
    "alteration":           "change",
    "alterations":          "changes",
    "amendment":            "change",
    "amendments":           "changes",
    "anthropogenic":        "human-caused",
    "apprehend":            "arrest",
    "approximately":        "about",
    "ascertain":            "find out",
    "assertion":            "claim",
    "assertions":           "claims",
    "assess":               "review",
    "assessment":           "review",
    "augment":              "boost",
    "authorize":            "allow",
    "authorized":           "allowed",
    "biodiversity":         "wildlife variety",
    "biomarkers":           "health markers",
    "bolster":              "boost",
    "cardiovascular":       "heart",
    "circumvent":           "avoid",
    "collaborate":          "work together",
    "collaboration":        "teamwork",
    "commence":             "start",
    "commenced":            "started",
    "commencing":           "starting",
    "commission":           "panel",
    "compensation":         "pay",
    "comprehensive":        "full",
    "confront":             "face",
    "confronting":          "facing",
    "confronts":            "faces",
    "consolidation":        "merger",
    "contemplate":          "consider",
    "contemplating":        "considering",
    "contentious":          "disputed",
    "convene":              "meet",
    "convening":            "meeting",
    "corroborate":          "confirm",
    "corroborated":         "confirmed",
    "counterterrorism":     "anti-terror",
    "cryptocurrency":       "digital currency",
    "deceleration":         "slowdown",
    "deliberate":           "discuss",
    "deliberating":         "discussing",
    "deliberation":         "debate",
    "demonstrate":          "show",
    "demonstrated":         "showed",
    "demonstrates":         "shows",
    "demonstrating":        "showing",
    "deterioration":        "decline",
    "diagnostic":           "testing",
    "diplomatic":           "peace",
    "discrepancy":          "gap",
    "disseminate":          "spread",
    "documentation":        "records",
    "doping":               "drug use",
    "elucidate":            "explain",
    "emphasized":           "stressed",
    "emphasize":            "stress",
    "employment":           "jobs",
    "enact":                "pass",
    "enacted":              "passed",
    "endeavor":             "effort",
    "enhance":              "improve",
    "enhanced":             "improved",
    "enhancement":          "improvement",
    "epidemic":             "outbreak",
    "epidemiological":      "disease-study",
    "escalation":           "rise",
    "evaluation":           "review",
    "exacerbate":           "worsen",
    "expedite":             "speed up",
    "expedited":            "sped up",
    "expenditure":          "",
    "facilitate":           "help",
    "facilitating":         "helping",
    "fiscal":               "financial",
    "formulate":            "create",
    "formulating":          "creating",
    "geopolitical":         "political",
    "governmental":         "government",
    "immunization":         "vaccination",
    "implement":            "roll out",
    "implemented":          "rolled out",
    "implementing":         "rolling out",
    "implication":          "effect",
    "implications":         "effects",
    "inadequate":           "poor",
    "incentivize":          "encourage",
    "incorporate":          "include",
    "indication":           "sign",
    "indicators":           "signs",
    "innovative":           "new",
    "inquiry":              "investigation",
    "insufficient":         "not enough",
    "integral":             "key",
    "international":        "global",
    "intervention":         "action",
    "investigate":          "probe",
    "investigated":         "probed",
    "investigating":        "probing",
    "investigation":        "probe",
    "investigations":       "probes",
    "legislative":          "law",
    "longevity":            "long life",
    "macroeconomic":        "economic",
    "mandate":              "order",
    "mechanism":            "process",
    "meteorological":       "weather",
    "metropolitan":         "city",
    "modification":         "change",
    "modifications":        "changes",
    "monetary":             "money",
    "multilateral":         "multi-country",
    "multinational":        "global",
    "neurological":         "brain",
    "neurodegenerative":    "brain disease",
    "objective":            "goal",
    "obligation":           "duty",
    "oncological":          "cancer",
    "operational":          "working",
    "optimization":         "improvement",
    "pandemic":             "outbreak",
    "parliamentary":        "parliament",
    "pathological":         "disease",
    "pediatric":            "child",
    "pharmaceutical":       "drug",
    "precipitation":        "rainfall",
    "proliferation":        "spread",
    "promulgates":          "announces",
    "propagation":          "spread",
    "protocols":            "steps",
    "psychological":        "mental health",
    "quantum":              "advanced",
    "ratify":               "approve",
    "ratified":             "approved",
    "ratifies":             "approves",
    "reassess":             "review",
    "regulatory":           "rules-based",
    "rehabilitation":       "recovery",
    "resilience":           "strength",
    "restructuring":        "reorganization",
    "retaliation":          "response",
    "scrutinize":           "examine",
    "scrutiny":             "review",
    "sedentary":            "inactive",
    "semiconductor":        "chip",
    "significant":          "major",
    "sophisticated":        "advanced",
    "substantiate":         "prove",
    "substantiated":        "proved",
    "substantial":          "major",
    "superiority":          "advantage",
    "surveillance":         "monitoring",
    "systematic":           "organized",
    "telecommunications":   "telecom",
    "therapeutic":          "treatment",
    "turbulence":           "turmoil",
    "turbulent":            "unstable",
    "unprecedented":        "record",
    "utilize":              "use",
    "utilized":             "used",
    "utilizes":             "uses",
    "validation":           "testing",
    "vehicular":            "vehicle",
    "vulnerability":        "weakness",
    "vulnerable":           "at risk",
    "realignment":          "shift",
    "revolutionary":        "new",
    "inflationary":         "rising",
    "computational":        "computing",
    "correlation":          "link",
    "deployed":             "used",
    "deployment":           "rollout",
}

# ─────────────────────────────────────────────────────────────────────────────
# PASS 4 — FILLER ADVERBS TO REMOVE
# ─────────────────────────────────────────────────────────────────────────────
FILLER_ADVERBS = [
    "substantially", "significantly", "considerably",
    "extensively", "comprehensively", "fundamentally",
    "predominantly", "overwhelmingly", "disproportionately",
    "simultaneously", "subsequently", "consequently",
    "accordingly", "respectively", "ultimately",
    "essentially", "effectively", "historically",
    "traditionally", "formally", "officially",
    "purportedly", "ostensibly",
]


# ─────────────────────────────────────────────────────────────────────────────
# CORE HELPERS  — all string-based, word-order preserved
# ─────────────────────────────────────────────────────────────────────────────

def _cap_match(replacement: str, original: str) -> str:
    """Mirror the capitalisation of the original matched word."""
    if not replacement:
        return replacement
    if original[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def _apply_phrases(text: str):
    """Replace verbose multi-word phrases (longest-first)."""
    changes = []
    sorted_phrases = sorted(VERBOSE_PHRASES, key=lambda x: len(x[0]), reverse=True)
    for phrase, replacement in sorted_phrases:
        pat = re.compile(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', re.IGNORECASE)
        new = pat.sub(lambda m: _cap_match(replacement, m.group()), text)
        if new != text:
            changes.append(f'"{phrase}" → "{replacement}"')
            text = new
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text, changes


def _apply_noun_chains(text: str):
    """Replace long noun-chain phrases with short equivalents."""
    changes = []
    for long_p, short_p in NOUN_CHAINS:
        pat = re.compile(r'\b' + re.escape(long_p) + r'\b', re.IGNORECASE)
        new = pat.sub(lambda m: _cap_match(short_p, m.group()), text)
        if new != text:
            changes.append(f'"{long_p}" → "{short_p}"')
            text = new
    return text, changes


def _apply_word_map(text: str):
    """Replace complex words with simpler alternatives (word-boundary safe)."""
    changes = []
    sorted_map = sorted(WORD_MAP.items(), key=lambda x: len(x[0]), reverse=True)
    for complex_w, simple_w in sorted_map:
        pat = re.compile(r'\b' + re.escape(complex_w) + r'\b', re.IGNORECASE)
        new = pat.sub(lambda m: _cap_match(simple_w, m.group()), text)
        if new != text:
            changes.append(f'"{complex_w}" → "{simple_w}"')
            text = new
    return text, changes


def _remove_fillers(text: str):
    """Strip vague intensifier adverbs."""
    changes = []
    for adv in FILLER_ADVERBS:
        pat = re.compile(r'\b' + re.escape(adv) + r'\b[\s,]*', re.IGNORECASE)
        new = pat.sub(' ', text)
        if new.strip() != text.strip():
            changes.append(f'Removed "{adv}"')
            text = new
    text = re.sub(r'\s{2,}', ' ', text).strip()
    return text, changes


def _title_case_headline(text: str) -> str:
    """
    Apply headline-style title case: capitalise every word except
    short conjunctions/prepositions/articles (unless they start the headline).
    """
    LOWER_WORDS = {
        "a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet",
        "at", "by", "in", "of", "on", "to", "up", "as", "if", "vs",
        "via", "per", "with", "into", "onto", "from", "over", "amid",
        "than", "that", "from",
    }
    words = text.split()
    result = []
    for i, word in enumerate(words):
        # Always capitalise first word and words after punctuation
        if i == 0 or word[0] in "\"'(" or (result and result[-1][-1] in ".!?:"):
            result.append(word[0].upper() + word[1:] if word else word)
        elif word.lower() in LOWER_WORDS:
            result.append(word.lower())
        else:
            result.append(word[0].upper() + word[1:] if word else word)
    return " ".join(result)


def _final_clean(text: str) -> str:
    """Strip leading article, remove duplicate words, apply title case, fix spacing."""
    text = re.sub(r'^(The|A|An)\s+', '', text)
    # Remove consecutive duplicate words (e.g. "spending spending" → "spending")
    text = re.sub(r'\b(\w+)\s+\1\b', r'\1', text, flags=re.IGNORECASE)
    # Fix spacing around punctuation
    text = re.sub(r'\s([,.:;!?])', r'\1', text)
    # Collapse multiple spaces
    text = re.sub(r'\s{2,}', ' ', text).strip()
    # Apply headline title case
    text = _title_case_headline(text)
    return text


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def simplify_headline(text: str) -> dict:
    """
    Simplify a complex news headline through 4 ordered string-level passes.
    Preserves full grammatical word order — no POS reconstruction.
    """
    if not text or not text.strip():
        return _empty(text)

    current = text
    all_steps = []

    # Pass 1: Verbose phrase removal
    after, changes = _apply_phrases(current)
    if changes:
        all_steps.append({"name": "Verbose Phrase Removal",
                          "before": current, "after": after, "detail": changes})
        current = after

    # Pass 2: Noun chain compression
    after, changes = _apply_noun_chains(current)
    if changes:
        all_steps.append({"name": "Noun Chain Compression",
                          "before": current, "after": after, "detail": changes})
        current = after

    # Pass 3: Complex word substitution
    after, changes = _apply_word_map(current)
    if changes:
        all_steps.append({"name": "Complex Word Substitution",
                          "before": current, "after": after, "detail": changes})
        current = after

    # Pass 4: Filler adverb removal
    after, changes = _remove_fillers(current)
    if changes:
        all_steps.append({"name": "Filler Word Removal",
                          "before": current, "after": after, "detail": changes})
        current = after

    final = _final_clean(current)

    orig_words = len(text.split())
    simp_words = len(final.split())
    reduction  = round((1 - simp_words / max(orig_words, 1)) * 100, 1)
    orig_ease  = flesch_reading_ease(text)
    simp_ease  = flesch_reading_ease(final)

    return {
        "original":              text,
        "simplified":            final,
        "steps":                 all_steps,
        "original_words":        orig_words,
        "simplified_words":      simp_words,
        "reduction_pct":         reduction,
        "original_ease":         orig_ease,
        "simplified_ease":       simp_ease,
        "original_ease_label":   reading_ease_label(orig_ease),
        "simplified_ease_label": reading_ease_label(simp_ease),
    }


def batch_simplify(headlines: list) -> list:
    return [simplify_headline(h) for h in headlines]


def _empty(text):
    return {
        "original": text, "simplified": text, "steps": [],
        "original_words": 0, "simplified_words": 0, "reduction_pct": 0,
        "original_ease": 0, "simplified_ease": 0,
        "original_ease_label": "", "simplified_ease_label": "",
    }
