# similarity_engine.py

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import re
import unicodedata
import sqlite3
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import jellyfish

# =========================
# GLOBALS
# =========================
SBERT_MODEL = None
PRGI_TITLES = []
PRGI_EMBEDDINGS = None

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'database', 'database.db')


# =========================
# INITIALIZATION
# =========================
def initialize_sbert_model():
    global SBERT_MODEL
    if SBERT_MODEL is None:
        print("✅ Loading SBERT model...")
        SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return SBERT_MODEL


# =========================
# TEXT NORMALIZATION
# =========================
def normalize_text(text: str) -> str:
    if not text:
        return ""

    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =========================
# LOAD DATASET
# =========================
def load_prgi_data():
    global PRGI_TITLES, PRGI_EMBEDDINGS

    try:
        from prgi_dataset import load_prgi_titles
        PRGI_TITLES = load_prgi_titles()

        print(f"✅ Loaded PRGI titles: {len(PRGI_TITLES)}")

        # 🔥 IMPORTANT
        initialize_sbert_model()
        PRGI_EMBEDDINGS = SBERT_MODEL.encode(PRGI_TITLES)

        print("🚀 PRGI embeddings precomputed")

    except Exception as e:
        print("❌ Dataset load error:", e)
        PRGI_TITLES = []
        PRGI_EMBEDDINGS = None


# =========================
# 🔥 HYBRID SIMILARITY
# =========================
def calculate_similarity(title1: str, title2: str) -> float:
    """
    Hybrid Similarity based on your paper:

    Hybrid = 0.2 * Phonetic + 0.3 * String + 0.5 * Semantic
    """

    global SBERT_MODEL

    if SBERT_MODEL is None:
        initialize_sbert_model()

    t1 = normalize_text(title1)
    t2 = normalize_text(title2)

    if not t1 or not t2:
        return 0.0

    # -------------------------
    # 1. Semantic (SBERT)
    # -------------------------
    emb1 = SBERT_MODEL.encode([t1])
    emb2 = SBERT_MODEL.encode([t2])
    semantic_score = cosine_similarity(emb1, emb2)[0][0]

    # -------------------------
    # 2. String (TF-IDF)
    # -------------------------
    tfidf = TfidfVectorizer().fit_transform([t1, t2])
    string_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    # -------------------------
    # 3. Phonetic (Metaphone)
    # -------------------------
    meta1 = jellyfish.metaphone(t1)
    meta2 = jellyfish.metaphone(t2)

    phonetic_score = 1.0 if meta1 == meta2 and meta1 != "" else 0.0

    # -------------------------
    # 🔥 HYBRID FORMULA
    # -------------------------
    final_score = (
        0.2 * phonetic_score +
        0.3 * string_score +
        0.5 * semantic_score
    )

    return round(final_score * 100, 2)


# =========================
# RISK CLASSIFICATION
# =========================
def classify_risk(score: float) -> str:
    if score >= 60:
        return "High"
    elif score >= 25:
        return "Medium"
    else:
        return "Low"


# =========================
# CORE FUNCTION (NO BREAK)
# =========================
def compare_title(user_title: str, prgi_titles=None):

    try:
        if not user_title:
            return {
                "similarity": 0,
                "risk": "Low",
                "closest_title": None
            }

        if prgi_titles is None:
            if not PRGI_TITLES:
                load_prgi_data()
            prgi_titles = PRGI_TITLES

        best_score = 0
        best_title = None

        # 🔥 PRECOMPUTE INPUT EMBEDDING ONCE
        t1 = normalize_text(user_title)
        input_emb = SBERT_MODEL.encode([t1])

        for i, title in enumerate(prgi_titles):

            t2 = normalize_text(title)

            # =========================
            # 🔥 FAST SEMANTIC (USING PRECOMPUTED EMBEDDINGS)
            # =========================
            semantic_score = cosine_similarity(
                input_emb,
                [PRGI_EMBEDDINGS[i]]
            )[0][0]

            # =========================
            # STRING (TF-IDF)
            # =========================
            tfidf = TfidfVectorizer().fit_transform([t1, t2])
            string_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

            # =========================
            # PHONETIC (Metaphone)
            # =========================
            meta1 = jellyfish.metaphone(t1)
            meta2 = jellyfish.metaphone(t2)
            phonetic_score = 1.0 if meta1 == meta2 and meta1 != "" else 0.0

            # =========================
            # 🔥 HYBRID SCORE
            # =========================
            score = (
                0.2 * phonetic_score +
                0.3 * string_score +
                0.5 * semantic_score
            ) * 100

            if score > best_score:
                best_score = score
                best_title = title

        risk = classify_risk(best_score)

        if best_score >= 60:
            explanation = "High similarity with existing PRGI titles."
        elif best_score >= 25:
            explanation = "Moderate similarity detected."
        else:
            explanation = "Title appears unique."

        return {
            "similarity": round(best_score, 2),
            "risk": risk,
            "closest_title": best_title,
            "explanation": explanation
        }

    except Exception as e:
        print("Error:", e)
        return {
            "similarity": 0,
            "risk": "Low",
            "closest_title": None
        }

# =========================
# INIT
# =========================
try:
    initialize_sbert_model()
    load_prgi_data()
except Exception as e:
    print("Startup error:", e)
    
def initialize_prgi_system():
    """
    🔥 Compatibility function (DO NOT REMOVE)

    Keeps your app.py working without changes
    """
    try:
        initialize_sbert_model()
        load_prgi_data()
        print("✅ PRGI system initialized")
    except Exception as e:
        print("Initialization error:", e)

def find_closest_title_match(input_title: str):
    """
    Compatibility function for existing app.py
    Uses new hybrid similarity internally
    """

    best_score = 0
    best_title = None
    best_scores = {}

    for title in PRGI_TITLES:
        scores = calculate_similarity(input_title, title)
        score = scores["hybrid"]

        if score > best_score:
            best_score = score
            best_title = title
            best_scores = scores

    return {
        "match": {"title": best_title},
        "scores": best_scores,
        "rank_score": best_score
    }

def get_titles_from_database():
    """
    Compatibility function for app.py
    Fetches titles from DB
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        cur.execute("SELECT title FROM submissions WHERE title IS NOT NULL")
        rows = cur.fetchall()

        conn.close()

        return [{"title": row[0]} for row in rows]

    except Exception as e:
        print("DB fetch error:", e)
        return []