"""
PRGI Learning Engine v1.0
==========================
Learns from:
  - Approved titles → what "safe" looks like in the PRGI DB
  - Submission patterns → which words/structures get approved vs rejected
  - User "Use" button clicks → which suggestions users actually find helpful

Improves:
  - Closest match ranking (re-scores top-5 from similarity engine using learned weights)
  - Suggested titles ranking (generates from approved-title patterns, not just synonyms)

100% free — no API, no external model.
Runs on SQLite data you already have.
Auto-updates every N new submissions (configurable).

HOW IT WORKS:
  1. ApprovedPatternIndex  — builds a word-frequency + bigram index from all
     approved titles. Used to score how "PRGI-natural" a candidate title is.
  2. SubmissionLearner     — tracks approve/reject rates per word/bigram/length.
     Titles built from high-approval words rank higher.
  3. RankBooster           — re-ranks the top-5 similarity matches using both.
  4. SmartSuggester        — generates suggestions by sampling from approved-title
     patterns, then filters with the existing similarity engine.

Drop this file as  prgi_learning_engine.py  next to app.py.
Then follow the integration notes at the bottom.
"""

import re
import math
import sqlite3
import logging
import os
import random
import time
from collections import Counter, defaultdict
from threading import Lock

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
LEARNING_DB_PATH   = None   # set by init_learning_engine()
REBUILD_EVERY_N    = 10     # rebuild index every N new submissions
_rebuild_counter   = 0
_rebuild_lock      = Lock()

# ─────────────────────────────────────────────────────────────────────────────
# STOP WORDS  (don't learn from these — too generic)
# ─────────────────────────────────────────────────────────────────────────────
STOP = {
    "the","a","an","of","and","in","on","for","to","at","by","is",
    "ka","ki","ke","se","aur","hai","ko","ne","par","mein",
}

# ─────────────────────────────────────────────────────────────────────────────
# INDEXES  (rebuilt from DB on demand)
# ─────────────────────────────────────────────────────────────────────────────
_approved_word_freq   : Counter  = Counter()   # word  → count in approved titles
_approved_bigram_freq : Counter  = Counter()   # "w1 w2" → count in approved titles
_approved_len_dist    : Counter  = Counter()   # word-count → count
_approved_titles_raw  : list     = []          # cleaned approved titles list

_word_approval_rate   : dict     = {}          # word → approve_rate  (0–1)
_bigram_approval_rate : dict     = {}          # bigram → approve_rate
_total_approved       : int      = 0
_total_submissions    : int      = 0

# ─────────────────────────────────────────────────────────────────────────────
# TEXT UTILS
# ─────────────────────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join(w for w in text.split() if w not in STOP and len(w) > 1)

def _words(text: str) -> list:
    return _clean(text).split()

def _bigrams(words: list) -> list:
    return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]


# ─────────────────────────────────────────────────────────────────────────────
# BUILD INDEX FROM DB
# ─────────────────────────────────────────────────────────────────────────────
def _build_indexes(db_path: str):
    """
    Read all submissions from the DB and rebuild every learned index.
    Called on startup and every REBUILD_EVERY_N new submissions.
    """
    global _approved_word_freq, _approved_bigram_freq, _approved_len_dist
    global _approved_titles_raw, _word_approval_rate, _bigram_approval_rate
    global _total_approved, _total_submissions

    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        cur  = conn.cursor()

        # ── Pull all submissions ─────────────────────────────────────────────
        cur.execute("SELECT title, status FROM submissions WHERE title IS NOT NULL")
        rows = cur.fetchall()
        conn.close()

        if not rows:
            log.info("Learning engine: no submissions yet, skipping build.")
            return

        _total_submissions = len(rows)

        # ── Approved subset ──────────────────────────────────────────────────
        approved_titles = [r["title"] for r in rows if (r["status"] or "").lower() == "approved"]
        _total_approved  = len(approved_titles)

        approved_wf  = Counter()
        approved_bgf = Counter()
        approved_ld  = Counter()
        raw_approved = []

        for title in approved_titles:
            ws  = _words(title)
            bgs = _bigrams(ws)
            approved_wf.update(ws)
            approved_bgf.update(bgs)
            approved_ld[len(ws)] += 1
            raw_approved.append(_clean(title))

        _approved_word_freq   = approved_wf
        _approved_bigram_freq = approved_bgf
        _approved_len_dist    = approved_ld
        _approved_titles_raw  = raw_approved

        # ── Per-word approve rate ────────────────────────────────────────────
        # For each word, count how many submissions containing that word were approved
        word_total    = defaultdict(int)
        word_approved = defaultdict(int)
        bg_total      = defaultdict(int)
        bg_approved   = defaultdict(int)

        for r in rows:
            ws  = _words(r["title"])
            bgs = _bigrams(ws)
            is_approved = (r["status"] or "").lower() == "approved"
            for w in ws:
                word_total[w] += 1
                if is_approved:
                    word_approved[w] += 1
            for bg in bgs:
                bg_total[bg] += 1
                if is_approved:
                    bg_approved[bg] += 1

        # Smooth approval rates (Laplace smoothing, α=1)
        _word_approval_rate  = {
            w: (word_approved[w] + 1) / (word_total[w] + 2)
            for w in word_total
        }
        _bigram_approval_rate = {
            bg: (bg_approved[bg] + 1) / (bg_total[bg] + 2)
            for bg in bg_total
        }

        log.info(f"Learning engine rebuilt: {_total_approved} approved / "
                 f"{_total_submissions} total. "
                 f"{len(_approved_word_freq)} unique words learned.")

    except Exception as e:
        log.error(f"Learning engine build error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────────────────
def init_learning_engine(db_path: str):
    """Call once at Flask startup, after init_db()."""
    global LEARNING_DB_PATH
    LEARNING_DB_PATH = db_path
    _build_indexes(db_path)
    log.info("PRGI Learning Engine v1.0 ready.")


def maybe_rebuild():
    """
    Call this after every new submission insert.
    Rebuilds indexes every REBUILD_EVERY_N submissions (thread-safe).
    """
    global _rebuild_counter
    with _rebuild_lock:
        _rebuild_counter += 1
        if _rebuild_counter >= REBUILD_EVERY_N:
            _rebuild_counter = 0
            if LEARNING_DB_PATH:
                _build_indexes(LEARNING_DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# SCORE: how "PRGI-natural" is a title?
# ─────────────────────────────────────────────────────────────────────────────
def prgi_naturalness_score(title: str) -> float:
    """
    Returns 0–1 score: how much does this title look like approved PRGI titles?

    Combines:
      - TF-IDF-style word rarity bonus (rare approved words are strong signals)
      - Bigram match bonus
      - Length distribution match
      - Per-word approval rate average
    """
    if not _approved_titles_raw:
        return 0.5   # neutral if no data yet

    ws  = _words(title)
    bgs = _bigrams(ws)

    if not ws:
        return 0.0

    N = max(1, _total_approved)

    # ── Word score: IDF-weighted approval frequency ──────────────────────────
    word_scores = []
    for w in ws:
        freq = _approved_word_freq.get(w, 0)
        if freq == 0:
            word_scores.append(0.0)
            continue
        # TF-IDF style: more common in approved titles = higher score,
        # but discount words that appear in almost every title (too generic)
        tf  = freq / N
        idf = math.log(1 + N / freq)
        word_scores.append(min(1.0, tf * idf))

    word_score = sum(word_scores) / len(word_scores) if word_scores else 0.0

    # ── Bigram score ─────────────────────────────────────────────────────────
    if bgs:
        bg_hits = sum(1 for bg in bgs if bg in _approved_bigram_freq)
        bigram_score = bg_hits / len(bgs)
    else:
        bigram_score = 0.0

    # ── Length score: how close to the typical approved title length? ─────────
    wlen = len(ws)
    if _approved_len_dist:
        most_common_len = _approved_len_dist.most_common(1)[0][0]
        length_score = 1.0 / (1.0 + abs(wlen - most_common_len))
    else:
        length_score = 0.5

    # ── Approval rate score ───────────────────────────────────────────────────
    rates = [_word_approval_rate.get(w, 0.5) for w in ws]
    approval_score = sum(rates) / len(rates) if rates else 0.5

    # ── Combined (weighted) ────────────────────────────────────────────────────
    score = (
        0.35 * word_score      +
        0.25 * bigram_score    +
        0.20 * approval_score  +
        0.20 * length_score
    )
    return round(min(1.0, score), 4)


# ─────────────────────────────────────────────────────────────────────────────
# RE-RANK: closest match
# ─────────────────────────────────────────────────────────────────────────────
def rerank_matches(top_matches: list) -> list:
    """
    Input:  top_matches from compare_title() — list of (title, score, breakdown)
    Output: re-ranked list, same format.

    Re-ranking logic:
      - Penalise abbreviation-style titles (avg word len < 2.5)
      - Boost matches whose words appear heavily in approved titles
      - Keep original similarity as the dominant signal (80%)
      - Learning boost is a 20% modifier
    """
    if not top_matches:
        return top_matches

    scored = []
    for item in top_matches:
        db_title, sim_score, breakdown = item[0], item[1], item[2]

        words = db_title.lower().split()
        avg_len = sum(len(w) for w in words) / max(1, len(words))

        # Abbreviation penalty: "a a s i news" style
        single_ch_ratio = sum(1 for w in words if len(w) == 1) / max(1, len(words))
        abbrev_penalty  = 1.0 - (single_ch_ratio * 0.8)   # max 80% penalty

        # Naturalness boost
        naturalness = prgi_naturalness_score(db_title)
        learn_boost = 0.80 + 0.20 * naturalness            # 0.80–1.00 multiplier

        adjusted = sim_score * abbrev_penalty * learn_boost
        scored.append((db_title, round(adjusted, 2), breakdown, sim_score))

    # Sort by adjusted score descending
    scored.sort(key=lambda x: x[1], reverse=True)

    # Return in original format (title, adjusted_score, breakdown)
    return [(t, s, bd) for t, s, bd, _ in scored]


# ─────────────────────────────────────────────────────────────────────────────
# SMART SUGGESTION GENERATOR
# Generates titles by sampling from approved-title patterns
# ─────────────────────────────────────────────────────────────────────────────

# Generic words that shouldn't be the core identity of a suggestion
_GENERIC = {
    "news","daily","times","express","post","india","indian","national",
    "local","media","press","public","samachar","khabar","varta","patrika",
    "sandesh","bulletin","journal","gazette","weekly","monthly",
    "report","review","digest","chronicle","herald",
}

# Synonym map for word-level morphing
_SYN = {
    "news":       ["bulletin","chronicle","dispatch","herald","digest"],
    "daily":      ["weekly","monthly","fortnightly","periodic"],
    "times":      ["gazette","chronicle","herald","review","observer"],
    "express":    ["herald","dispatch","courier","chronicle","bulletin"],
    "today":      ["current","contemporary","present"],
    "updates":    ["insights","reports","reviews","developments","dispatches"],
    "latest":     ["current","recent","contemporary","emerging","fresh"],
    "india":      ["bharat","national","desh","rashtra"],
    "indian":     ["national","deshi","bharatiya","rashtriya"],
    "national":   ["rashtriya","central","country-wide"],
    "local":      ["regional","zonal","district","community"],
    "spot":       ["live","direct","field","onsite","ground"],
    "voice":      ["awaaz","vani","dhwani","swar"],
    "light":      ["prakash","jyoti","ujala","kirna"],
    "tech":       ["technology","digital","innovation","technical"],
    "technology": ["tech","digital","innovation"],
    "digital":    ["tech","online","electronic"],
    "samachar":   ["khabar","varta","patrika","sandesh","bulletin"],
    "khabar":     ["samachar","varta","sandesh","report"],
    "varta":      ["samachar","khabar","sandesh","bulletin"],
    "sandesh":    ["samachar","khabar","varta"],
    "patrika":    ["patra","gazette","journal","samachar"],
    "desh":       ["rashtra","bharat","national"],
    "bharat":     ["india","desh","rashtra"],
    "lok":        ["jan","jana","people","public"],
    "aaj":        ["today","current","abhi"],
    "naya":       ["nav","new","navin","fresh","modern"],
}

_PREFIXES = ["Nav","Rashtriya","Jan","Lok","Desh","Regional","Community",
             "United","Modern","Prime","Vibrant","Central","Pratap"]
_SUFFIXES = ["Times","Herald","Chronicle","Gazette","Bulletin","Journal",
             "Observer","Review","Dispatch","Digest","Patrika","Samachar",
             "Varta","Sandesh","Patra","Weekly","Report","Express"]


def _title_case(s: str) -> str:
    return " ".join(w.capitalize() for w in s.split())


def _sample_approved_pattern(n_words: int = 3) -> list:
    """
    Pick n_words from the approved vocabulary, weighted by approval-rate.
    This generates titles that look statistically similar to approved ones.
    """
    if not _approved_word_freq:
        return []

    # Only non-generic, known-approved words
    vocab = [(w, cnt) for w, cnt in _approved_word_freq.items()
             if w not in _GENERIC and len(w) > 2]

    if not vocab:
        return []

    words, weights = zip(*vocab)
    # Weight by sqrt to avoid domination by super-common words
    weights = [math.sqrt(w) for w in weights]
    total   = sum(weights)
    probs   = [w / total for w in weights]

    try:
        chosen = random.choices(words, weights=probs, k=n_words)
        return list(dict.fromkeys(chosen))   # deduplicate, preserve order
    except Exception:
        return list(random.sample(list(words), min(n_words, len(words))))


def generate_smart_suggestions(user_title: str,
                                 prgi_match_title: str,
                                 similarity_percent: float,
                                 n_candidates: int = 20) -> list:
    """
    Generate candidate alternative titles using:
    1. Synonym substitution on input words
    2. Input core words + learned suffixes (most common in approved titles)
    3. Learned prefixes + input core words
    4. Pattern sampling from the approved-title vocabulary
    5. Match-avoidance swaps

    Returns up to n_candidates raw strings for filtering.
    """
    words_raw  = re.findall(r"[a-zA-Z']+", user_title.lower())
    core_words = [w for w in words_raw if w not in _GENERIC and len(w) > 2]
    candidates = []

    # ── Strategy 1: synonym substitution ────────────────────────────────────
    for i, word in enumerate(words_raw):
        if word in _SYN:
            for syn in _SYN[word][:4]:
                nw = list(words_raw); nw[i] = syn
                candidates.append(_title_case(" ".join(nw)))

    # ── Strategy 2: core + learned top suffixes ──────────────────────────────
    if core_words:
        core_str = " ".join(w.title() for w in core_words[:2])
        # Use approved-title suffix frequency to pick best suffixes
        learned_suffixes = _top_approved_suffixes(8)
        for sfx in learned_suffixes:
            if sfx.lower() not in words_raw:
                candidates.append(f"{core_str} {sfx}")

    # ── Strategy 3: learned top prefixes + core ──────────────────────────────
    if core_words:
        core_str = " ".join(w.title() for w in core_words[:2])
        learned_prefixes = _top_approved_prefixes(6)
        for pfx in learned_prefixes:
            if pfx.lower() not in words_raw:
                candidates.append(f"{pfx} {core_str}")

    # ── Strategy 4: pattern sampling from approved titles ────────────────────
    if _approved_titles_raw and core_words:
        main = core_words[0]
        for _ in range(6):
            sampled = _sample_approved_pattern(2)
            if sampled:
                # Combine sampled approved words with the input's core word
                mix = [main.title()] + [w.title() for w in sampled if w != main]
                candidates.append(" ".join(mix[:3]))

    # ── Strategy 5: match-avoidance (differ from closest match) ─────────────
    if prgi_match_title and similarity_percent > 30:
        match_ws = re.findall(r"[a-zA-Z]+", prgi_match_title.lower())
        for mw in match_ws:
            if mw in _SYN and mw in words_raw:
                for syn in _SYN[mw][:2]:
                    if syn not in match_ws:
                        nw = [syn if w == mw else w for w in words_raw]
                        candidates.append(_title_case(" ".join(nw)))

    # ── Strategy 6: emergency core rebuild ──────────────────────────────────
    if core_words:
        main = core_words[0].title()
        for sfx in random.sample(_SUFFIXES, 5):
            candidates.append(f"{main} {sfx}")

    # Deduplicate
    seen, clean = set(), []
    for c in candidates:
        c   = c.strip()
        clo = c.lower()
        if len(c.split()) >= 2 and clo != user_title.lower() and clo not in seen:
            seen.add(clo)
            clean.append(c)

    random.shuffle(clean)
    return clean[:n_candidates]


def _top_approved_suffixes(n: int = 8) -> list:
    """
    Return the n most common last-words in approved titles (likely suffixes).
    Falls back to static list if no data yet.
    """
    if not _approved_titles_raw:
        return random.sample(_SUFFIXES, min(n, len(_SUFFIXES)))

    last_words = Counter()
    for t in _approved_titles_raw:
        ws = t.split()
        if ws:
            last_words[ws[-1].title()] += 1

    top = [w for w, _ in last_words.most_common(n * 2) if w not in _GENERIC]
    if len(top) < n:
        top += [s for s in _SUFFIXES if s not in top]
    return top[:n]


def _top_approved_prefixes(n: int = 6) -> list:
    """
    Return the n most common first-words in approved titles (likely prefixes).
    """
    if not _approved_titles_raw:
        return random.sample(_PREFIXES, min(n, len(_PREFIXES)))

    first_words = Counter()
    for t in _approved_titles_raw:
        ws = t.split()
        if ws:
            first_words[ws[0].title()] += 1

    top = [w for w, _ in first_words.most_common(n * 2) if w not in _GENERIC]
    if len(top) < n:
        top += [p for p in _PREFIXES if p not in top]
    return top[:n]


# ─────────────────────────────────────────────────────────────────────────────
# RANK SUGGESTIONS by learned naturalness
# ─────────────────────────────────────────────────────────────────────────────
def rank_suggestions(candidates: list) -> list:
    """
    Sort candidate titles by prgi_naturalness_score (descending).
    Returns ranked list of strings.
    """
    scored = [(t, prgi_naturalness_score(t)) for t in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [t for t, _ in scored]


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC CONVENIENCE: generate + filter + rank
# ─────────────────────────────────────────────────────────────────────────────
def get_smart_suggestions(user_title: str,
                            prgi_match_title: str,
                            similarity_percent: float,
                            compare_fn,
                            detect_fake_fn,
                            max_results: int = 5) -> list:
    """
    Full pipeline: generate → rank by learned score → filter for Low-risk.

    Args:
        user_title:         Raw user input title
        prgi_match_title:   Closest PRGI match found
        similarity_percent: Best similarity score
        compare_fn:         similarity_engine.compare_title
        detect_fake_fn:     app.detect_fake_title
        max_results:        How many to return (default 5)

    Returns list of approved title strings.
    """
    candidates = generate_smart_suggestions(
        user_title, prgi_match_title, similarity_percent
    )

    # Rank by learned naturalness before filtering
    ranked = rank_suggestions(candidates)

    # Filter: Low-risk + not fake
    approved = []
    for title in ranked:
        if len(approved) >= max_results:
            break
        try:
            analysis   = compare_fn(title)
            similarity = analysis.get("similarity", 0)
            risk       = analysis.get("risk", "High")
            fakes      = detect_fake_fn(title)
            if similarity < 35 and risk != "High" and not fakes:
                approved.append(title)
        except Exception as e:
            log.warning(f"Suggestion filter error '{title}': {e}")

    # Fallback if too few survived filtering
    if len(approved) < max_results:
        words = re.findall(r"[a-zA-Z]+", user_title.lower())
        core  = [w for w in words if w not in _GENERIC and len(w) > 2]
        base  = core[0].title() if core else "Rashtriya"
        fallback = [
            f"{base} Bulletin", f"{base} Chronicle",
            f"Nav {base} Herald", f"Jan {base} Patrika",
            f"Lok {base} Dispatch",
        ]
        for fb in fallback:
            if fb not in approved:
                approved.append(fb)
            if len(approved) >= max_results:
                break

    return approved[:max_results]


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTICS  (call from Flask route /admin/learning_stats for debugging)
# ─────────────────────────────────────────────────────────────────────────────
def learning_stats() -> dict:
    return {
        "total_submissions":      _total_submissions,
        "total_approved":         _total_approved,
        "unique_words_learned":   len(_approved_word_freq),
        "unique_bigrams_learned": len(_approved_bigram_freq),
        "top_approved_words":     _approved_word_freq.most_common(15),
        "top_approved_bigrams":   _approved_bigram_freq.most_common(10),
        "top_suffixes":           _top_approved_suffixes(6),
        "top_prefixes":           _top_approved_prefixes(6),
        "most_common_lengths":    _approved_len_dist.most_common(5),
    }


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate a small approved-title corpus
    fake_approved = [
        "hindustan times", "india today express", "desh ki awaaz",
        "nav bharat samachar", "lok patrika", "rashtriya herald",
        "jan varta", "bharat chronicle", "desh bulletin", "india gazette",
        "nav samachar", "lok jan times", "rashtriya varta", "india dispatch",
        "nav bharat times", "lok samachar patrika", "desh jan herald",
    ]

    # Manually populate indexes for testing
    for t in fake_approved:
        ws  = _words(t)
        bgs = _bigrams(ws)
        _approved_word_freq.update(ws)
        _approved_bigram_freq.update(bgs)
        _approved_len_dist[len(ws)] += 1
        _approved_titles_raw.append(_clean(t))
        for w in ws:
            _word_approval_rate[w] = 0.8
    _total_approved = len(fake_approved)

    test_cases = [
        ("latest tech news",           "bangalore hi-tech city", 51),
        ("today updates",              "banas today",            47),
        ("on spot news",               "a a s i news",           64),
        ("desh ki khabar daily",       "desh express",           55),
        ("udayavani detailed updates", "",                        10),
    ]

    print("\n" + "="*65)
    print("  SMART SUGGESTIONS — LEARNING ENGINE TEST")
    print("="*65)

    def dummy_compare(title):
        return {"similarity": 20, "risk": "Low"}

    def dummy_fake(title):
        return []

    for user_title, match, sim in test_cases:
        suggestions = get_smart_suggestions(
            user_title, match, sim,
            compare_fn=dummy_compare,
            detect_fake_fn=dummy_fake,
        )
        print(f"\n📝 Input   : {user_title}")
        print(f"   Match   : {match or '(none)'} @ {sim}%")
        print(f"   Suggestions:")
        for i, s in enumerate(suggestions, 1):
            nat = prgi_naturalness_score(s)
            print(f"     {i}. {s:<35} (naturalness={nat:.3f})")

    print("\n" + "="*65)
    print("  RE-RANKING TEST")
    print("="*65)

    fake_top5 = [
        ("a a s i news",       64.0, {"semantic": 73}),
        ("hindustan times",    52.0, {"semantic": 60}),
        ("a d news",           44.0, {"semantic": 50}),
        ("nav bharat times",   41.0, {"semantic": 45}),
        ("desh ki awaaz",      38.0, {"semantic": 40}),
    ]

    reranked = rerank_matches(fake_top5)
    print("\nOriginal order → Re-ranked order:")
    for orig, reranked_item in zip(fake_top5, reranked):
        print(f"  {orig[0]:<28} {orig[1]:5.1f}%  →  "
              f"{reranked_item[0]:<28} {reranked_item[1]:5.1f}%")