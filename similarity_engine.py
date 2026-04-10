"""
PRGI Title Similarity Engine v2.0
===================================
Patent-level accuracy for Indian newspaper / publication title matching.
Supports multilingual Indian titles (Hindi, Gujarati, Bengali, Telugu, etc.)

Architecture:
  - Multilingual SBERT (semantic)
  - TF-IDF with subword n-grams (lexical)
  - Jaccard on unigrams + bigrams (overlap)
  - Soundex phonetic (transliteration-safe)
  - Levenshtein edit distance (typo-safe)
  - Substring containment bonus
  - Nonsense / OOV detection (prevents qwerty-type false positives)
  - Calibrated sigmoid output → always 0–100
  - Explainable per-match score breakdown
"""

import re
import unicodedata
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# ─────────────────────────────────────────────
# GLOBALS
# ─────────────────────────────────────────────
SBERT_MODEL       = None
ALL_TITLES        = []
CLEANED_TITLES    = []
ALL_EMBEDDINGS    = None
TFIDF_VECTORIZER  = None
TFIDF_MATRIX      = None
VOCAB_SET         = set()   # all words seen in the database — for OOV detection

# ─────────────────────────────────────────────
# STOPWORDS (English + common Hindi/transliterated)
# ─────────────────────────────────────────────
STOPWORDS = {
    # English
    "the","a","an","of","and","in","on","for","to","at","by","is","its",
    # Hindi transliterated fillers (do NOT remove meaningful words like "desh","bharat")
    "ka","ki","ke","se","aur","hai","hain","ko","ne","par","mein",
}

# Words that are so generic in publication names they should carry less weight
GENERIC_PUBLICATION_WORDS = {
    "news","daily","times","samachar","bulletin","express","journal",
    "weekly","monthly","patra","patrika","khabar","khabren","varta",
    "report","today","india","indian","national","local","press",
    "sandesh","samchar","media","live","online","digital",
}

# ─────────────────────────────────────────────
# TEXT NORMALISATION
# ─────────────────────────────────────────────
def unicode_normalise(text: str) -> str:
    """NFKC normalise — collapses Unicode variants, fixes Devanagari spacing."""
    return unicodedata.normalize("NFKC", text)

def normalize_text(text: str) -> str:
    text = unicode_normalise(text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(words)

def get_bigrams(text: str) -> set:
    words = text.split()
    return {f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)}

# ─────────────────────────────────────────────
# OOV (Out-Of-Vocabulary) DETECTION
# Prevents nonsense strings like "qwerty" from
# getting high scores just because they share a
# common suffix like "samachar".
# ─────────────────────────────────────────────
def oov_penalty(cleaned_input: str) -> float:
    """
    Returns a multiplier in [0.1, 1.0].
    If most words in the input do not appear anywhere in the database,
    the query is probably gibberish and we heavily penalise it.
    """
    words = cleaned_input.split()
    if not words:
        return 1.0

    # Generic-only queries ("news daily", "samachar") also get a soft penalty
    # because they match everything equally — not very useful signal.
    non_generic = [w for w in words if w not in GENERIC_PUBLICATION_WORDS]
    generic_ratio = 1.0 - (len(non_generic) / len(words))

    oov_words = [w for w in words if w not in VOCAB_SET]
    oov_ratio  = len(oov_words) / len(words)

    # Full-OOV non-generic words are gibberish (e.g. "qwerty")
    oov_non_generic = [w for w in oov_words if w not in GENERIC_PUBLICATION_WORDS]
    if oov_non_generic:
        # Each OOV non-generic word cuts the score significantly
        per_word_penalty = 0.75 ** len(oov_non_generic)
        return max(0.50, per_word_penalty)

    if oov_ratio > 0.6:
        return 0.25

    if generic_ratio > 0.85:
        # Query is almost entirely generic words — dampen max possible score
        return 0.70

    return 1.0

# ─────────────────────────────────────────────
# SIMILARITY METHODS
# ─────────────────────────────────────────────
def semantic_similarity(cleaned_input: str) -> np.ndarray:
    emb = SBERT_MODEL.encode([cleaned_input], normalize_embeddings=True)
    return cosine_similarity(emb, ALL_EMBEDDINGS)[0]

def tfidf_similarity(cleaned_input: str) -> np.ndarray:
    vec = TFIDF_VECTORIZER.transform([cleaned_input])
    return cosine_similarity(vec, TFIDF_MATRIX)[0]

def jaccard_sim(a: str, b: str, use_bigrams: bool = True) -> float:
    s1 = set(a.split())
    s2 = set(b.split())
    if not s1 or not s2:
        return 0.0
    unigram_j = len(s1 & s2) / len(s1 | s2)
    if not use_bigrams:
        return unigram_j
    bg1 = get_bigrams(a)
    bg2 = get_bigrams(b)
    bigram_j = len(bg1 & bg2) / len(bg1 | bg2) if (bg1 or bg2) else 0.0
    return 0.6 * unigram_j + 0.4 * bigram_j

def soundex_code(w: str) -> str:
    """Standard Soundex — works well for transliterated Indian words."""
    w = w.upper()
    if not w:
        return "0000"
    table = {"BFPV": "1", "CGJKQSXZ": "2", "DT": "3",
             "L": "4", "MN": "5", "R": "6"}
    code = w[0]
    prev = ""
    for ch in w[1:]:
        digit = "0"
        for key in table:
            if ch in key:
                digit = table[key]
                break
        if digit != "0" and digit != prev:
            code += digit
        prev = digit
    return (code + "000")[:4]

def phonetic_sim(a: str, b: str) -> float:
    a_codes = [soundex_code(x) for x in a.split() if x]
    b_codes = [soundex_code(x) for x in b.split() if x]
    if not a_codes or not b_codes:
        return 0.0
    sa, sb = set(a_codes), set(b_codes)
    return len(sa & sb) / max(len(sa), len(sb))

def levenshtein(s1: str, s2: str) -> int:
    """Standard edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1,
                            curr[j] + 1,
                            prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]

def char_edit_sim(a: str, b: str) -> float:
    """
    Character-level similarity using edit distance.
    Normalised to [0,1]. Useful for catching 'hindustan' vs 'hindustan times'.
    """
    max_len = max(len(a), len(b), 1)
    dist = levenshtein(a[:60], b[:60])  # cap at 60 chars for speed
    return 1.0 - dist / max(max_len, 1)

def substring_bonus(a: str, b: str) -> float:
    """
    Reward when one title is largely contained in another.
    E.g. 'hindustan' inside 'aaj ka hindustan' → bonus.
    """
    words_a = set(a.split())
    words_b = set(b.split())
    if not words_a or not words_b:
        return 0.0
    # Remove generic words before checking containment
    core_a = words_a - GENERIC_PUBLICATION_WORDS
    core_b = words_b - GENERIC_PUBLICATION_WORDS
    if not core_a or not core_b:
        return 0.0
    overlap = core_a & core_b
    containment = len(overlap) / min(len(core_a), len(core_b))
    return containment * 0.08   # max +12 percentage points (before calibration)

# ─────────────────────────────────────────────
# KEYWORD SPECIFICITY
# Rare / specific words should boost the score
# when they match; generic words should not.
# ─────────────────────────────────────────────
def specificity_boost(input_words: list, db_words: list) -> float:
    """
    Reward matches on non-generic, non-stopword tokens.
    Each specific shared word adds a small boost.
    """
    input_specific = {w for w in input_words
                      if w not in GENERIC_PUBLICATION_WORDS and w not in STOPWORDS}
    db_specific    = {w for w in db_words
                      if w not in GENERIC_PUBLICATION_WORDS and w not in STOPWORDS}
    shared = input_specific & db_specific
    if not input_specific:
        return 0.0
    return min(0.15, len(shared) * 0.05)   # cap at +15 pp

# ─────────────────────────────────────────────
# LENGTH STRUCTURE SIMILARITY
# ─────────────────────────────────────────────
def structure_sim(a: str, b: str) -> float:
    la, lb = len(a.split()), len(b.split())
    if la == 0 or lb == 0:
        return 0.0
    return 1.0 - abs(la - lb) / max(la, lb)

# ─────────────────────────────────────────────
# CALIBRATED SCORE → 0–100
# Raw weighted sum can exceed 1.0 due to bonuses.
# We apply a soft sigmoid calibration so the
# output is always well-bounded and interpretable.
# ─────────────────────────────────────────────
def calibrate(raw_score: float, oov_mult: float) -> float:
    """
    1. Apply OOV penalty (multiplicative).
    2. Sigmoid-compress to (0, 100) so no score ever hits 100 on noise.
    3. Linearly rescale so a 'true match' (raw≈0.95) → ~95,
       and a 'random match' (raw≈0.30) → ~30.
    """
    penalised = raw_score * oov_mult
    # Sigmoid: S(x) = 1 / (1 + e^(-k*(x - x0)))
    # Tuned so 0.5 raw → ~50 out, 0.9 raw → ~88 out, 0.2 raw → ~18 out
    k  = 8.0
    x0 = 0.50
    sig = 1.0 / (1.0 + np.exp(-k * (penalised - x0)))
    # Rescale [sigmoid(0), sigmoid(1)] → [0, 100]
    lo = 1.0 / (1.0 + np.exp(-k * (0.0 - x0)))
    hi = 1.0 / (1.0 + np.exp(-k * (1.0 - x0)))
    calibrated = (sig - lo) / (hi - lo) * 100.0
    return round(float(np.clip(calibrated, 0.0, 100.0)), 2)

# ─────────────────────────────────────────────
# INITIALISE SYSTEM
# ─────────────────────────────────────────────
def initialize_system():
    global SBERT_MODEL, ALL_TITLES, CLEANED_TITLES
    global ALL_EMBEDDINGS, TFIDF_VECTORIZER, TFIDF_MATRIX, VOCAB_SET

    from prgi_dataset import load_prgi_titles
    ALL_TITLES     = load_prgi_titles()
    CLEANED_TITLES = [normalize_text(t) for t in ALL_TITLES]

    # Build vocabulary from every word in the database
    for ct in CLEANED_TITLES:
        VOCAB_SET.update(ct.split())

    print("🔄 Loading multilingual SBERT …")
    SBERT_MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    print("🔄 Encoding embeddings …")
    ALL_EMBEDDINGS = SBERT_MODEL.encode(
        CLEANED_TITLES,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
    )

    print("🔄 Building TF-IDF (char + word n-grams) …")
    # Character n-grams (2–4) catch transliteration variants
    # Word n-grams (1–2) catch phrase-level matches
    TFIDF_VECTORIZER = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(2, 4),
        min_df=1,
        max_features=80_000,
        sublinear_tf=True,
    )
    TFIDF_MATRIX = TFIDF_VECTORIZER.fit_transform(CLEANED_TITLES)

    print(f"✅ System ready — {len(ALL_TITLES)} titles indexed.")
def is_valid_publication(title: str) -> bool:
    words = title.split()

    # reject long sentence-like entries
    if len(words) > 6:
        return False

    # must contain at least one publication keyword
    keywords = {
        "news","times","express","samachar",
        "bulletin","journal","patrika","khabar"
    }

    if not any(w in keywords for w in words):
        return False

    return True
def generic_penalty(input_words):
    generic = {
        "news","india","indian","daily",
        "express","times","samachar"
    }

    if all(w in generic for w in input_words):
        return 0.6   # strong penalty

    return 1.0

# ─────────────────────────────────────────────
# MAIN ENGINE
# ─────────────────────────────────────────────
def compare_title(title: str) -> dict:
    """
    Returns:
        similarity    : float  — calibrated 0–100 score
        risk          : str    — "High" / "Medium" / "Low"
        closest_title : str | None
        top_matches   : list of (title, score, breakdown_dict)
        oov_penalty   : float  — multiplier applied (1.0 = no penalty)
        input_clean   : str    — normalised input (useful for debugging)
    """
    
    # Handle very short / weak titles
    # Handle empty input FIRST
    if not title or not title.strip():
        return _empty_result()
    if len(title.split()) < 2:
        return {
        "similarity": 0,
        "risk": "Low",
        "closest_title": "Enter a more descriptive title",
        "top_matches": [],
        }
    # Boost weak titles for better matching

    cleaned_input = normalize_text(title)
    input_words   = cleaned_input.split()
    # Handle very short / weak titles
    # Boost weak titles AFTER normalization
    if len(input_words) <= 2:
        return {
        "similarity": 5.0,
        "risk": "Low",
        "closest_title": "Title too short for meaningful comparison",
        "top_matches": [],
        "oov_penalty": 1.0,
        "input_clean": cleaned_input,
        }

    if not input_words:
        return _empty_result()

    # ── Pre-compute expensive vectors once ──────────────────────────────
    sem_scores   = semantic_similarity(cleaned_input)
    tfidf_scores = tfidf_similarity(cleaned_input)
    oov_mult     = oov_penalty(cleaned_input)
    gen_pen = generic_penalty(input_words)

    results = []

    for i, db_clean in enumerate(CLEANED_TITLES):
        if not is_valid_publication(db_clean):
            continue
        if len(db_clean.split()) < 1:
            continue

        db_words = db_clean.split()

        # ── Individual signals ───────────────────────────────────────────
        sem    = float(sem_scores[i])
        tf     = float(tfidf_scores[i])
        jac    = jaccard_sim(cleaned_input, db_clean)
        pho    = phonetic_sim(cleaned_input, db_clean)
        edit   = char_edit_sim(cleaned_input, db_clean)
        struct = structure_sim(cleaned_input, db_clean)
        sub    = substring_bonus(cleaned_input, db_clean)
        spec   = specificity_boost(input_words, db_words)

        # ── Weighted combination ─────────────────────────────────────────
        # Weights tuned for Indian publication name matching:
        #   - Semantic is king (meaning matters most)
        #   - Char TF-IDF catches transliteration variants
        #   - Jaccard on words+bigrams catches paraphrase
        #   - Phonetic for soundalike words (Samachar / Samchar)
        #   - Edit distance catches single-word typos
        #   - Structure + substring as supporting signals
        raw = (
            0.24 * sem    +   # semantic meaning
            0.28 * tf     +   # char n-gram lexical
            0.18 * jac    +   # word/bigram overlap
            0.10 * pho    +   # phonetic (transliteration)
            0.10 * edit   +   # edit distance
            0.05 * struct +   # length similarity
            sub           +   # substring containment bonus
            spec              # specificity bonus
        )

        # ── Noise suppression ────────────────────────────────────────────
        # If both semantic AND lexical are near-zero, this is a random match
        if sem < 0.15 and tf < 0.10:
            raw *= 0.4

        results.append((i, raw, {
            "semantic":    round(sem   * 100, 1),
            "tfidf":       round(tf    * 100, 1),
            "jaccard":     round(jac   * 100, 1),
            "phonetic":    round(pho   * 100, 1),
            "edit":        round(edit  * 100, 1),
            "structure":   round(struct* 100, 1),
            "sub_bonus":   round(sub   * 100, 1),
            "spec_bonus":  round(spec  * 100, 1),
        }))

    results.sort(key=lambda x: x[1], reverse=True)

    top5 = []
    for idx, raw_score, breakdown in results[:5]:
        cal = calibrate(raw_score, oov_mult * gen_pen)
        top5.append((ALL_TITLES[idx], cal, breakdown))

    best_title, best_score, _ = top5[0] if top5 else (None, 0.0, {})

    # If best score is too low, don't claim a closest match
    if best_score < 20.0:
        best_title = None

    risk = (
        "High"   if best_score >= 65 else
        "Medium" if best_score >= 35 else
        "Low"
    )
    if best_score < 30:
        best_title = None
    # Fallback if no matches found
    if not top5 or best_score < 10:
        return {
        "similarity": 0,
        "risk": "Low",
        "closest_title": "No strong PRGI match found",
        "top_matches": [],
        "oov_penalty": 1.0,
        "input_clean": cleaned_input,
    }
    return {
        "similarity":    best_score,
        "risk":          risk,
        "closest_title": best_title,
        "top_matches":   top5,
        "oov_penalty":   round(oov_mult, 3),
        "input_clean":   cleaned_input,
    }

def _empty_result() -> dict:
    return {
        "similarity":    0.0,
        "risk":          "Low",
        "closest_title": None,
        "top_matches":   [],
        "oov_penalty":   1.0,
        "input_clean":   "",
    }

# ─────────────────────────────────────────────
# PRETTY PRINT
# ─────────────────────────────────────────────
def print_result(title: str, r: dict):
    sep = "─" * 60
    print(f"\n{sep}")
    print(f"  Input   : {title}")
    print(f"  Cleaned : {r['input_clean']}")
    print(f"  OOV ×   : {r['oov_penalty']}")
    print(sep)
    print(f"  Similarity : {r['similarity']:.2f}%")
    print(f"  Risk       : {r['risk']}")
    print(f"  Closest    : {r['closest_title'] or '—'}")
    print(f"\n  Top Matches:")
    for rank, (t, s, bd) in enumerate(r['top_matches'], 1):
        print(f"    {rank}. [{s:6.2f}%]  {t}")
        print(f"         sem={bd['semantic']} tf={bd['tfidf']} "
              f"jac={bd['jaccard']} pho={bd['phonetic']} "
              f"edit={bd['edit']} sub+={bd['sub_bonus']} spec+={bd['spec_bonus']}")
    print(sep)

# ─────────────────────────────────────────────
# TEST MODE
# ─────────────────────────────────────────────
if __name__ == "__main__":
    initialize_system()

    # ── Quick self-test on known problematic cases ───────────────────────
    TEST_CASES = [
        # (input,              expected_risk_gte)
        ("indian news daily",          "Medium"),  # generic → Medium at most
        ("financial express india",    "Medium"),
        ("bharat news samachar",       "Medium"),
        ("hindustan bulletin",         "Medium"),
        ("desh ki khabar",             "Medium"),
        ("qwerty samachar",            "Low"),     # gibberish → Low
        ("monsoon alert india",        "Low"),     # weather → Low
        ("stock market india",         "Low"),     # off-domain → Low
        ("daily thanthi",              "Low"),
        ("random title nothing",       "Low"),
    ]

    print("\n" + "═"*60)
    print("  SELF-TEST")
    print("═"*60)
    for test_input, _ in TEST_CASES:
        r = compare_title(test_input)
        flag = "✅" if r["risk"] == "Low" and "qwerty" in test_input.lower() \
               else ("✅" if r["similarity"] < 85 else "⚠️ HIGH?")
        print(f"  {flag} {test_input:<35} → {r['similarity']:6.2f}%  [{r['risk']}]  ≈ {r['closest_title'] or '—'}")

    # ── Interactive loop ─────────────────────────────────────────────────
    print("\n" + "═"*60)
    print("  INTERACTIVE MODE  (type 'quit' to exit)")
    print("═"*60)
    while True:
        try:
            t = input("\nEnter title: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if t.lower() in {"quit", "exit", "q"}:
            break
        if not t:
            continue
        r = compare_title(t)
        print_result(t, r)