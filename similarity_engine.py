# similarity_engine.py

import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import re
import unicodedata
import sqlite3
import pandas as pd
from typing import List

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import jellyfish

# =========================
# GLOBALS
# =========================
SBERT_MODEL = None
ALL_TITLES = []
ALL_EMBEDDINGS = None

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'database', 'database.db')


# =========================
# INIT MODEL
# =========================
def initialize_sbert_model():
    global SBERT_MODEL
    if SBERT_MODEL is None:
        print("✅ Loading SBERT model...")
        SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return SBERT_MODEL


# =========================
# LOAD CLEAN DATASET
# =========================
def load_clean_titles():
    try:
        df = pd.read_csv("database/cleaned_titles.csv")
        titles = df["title"].dropna().tolist()
        print(f"✅ Loaded cleaned titles: {len(titles)}")
        return titles
    except Exception as e:
        print("❌ Error loading cleaned dataset:", e)
        return []


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
# INITIALIZE SYSTEM
# =========================
def initialize_system():
    global ALL_TITLES, ALL_EMBEDDINGS

    initialize_sbert_model()

    ALL_TITLES = load_clean_titles()

    print("🚀 Generating embeddings (one-time)...")
    ALL_EMBEDDINGS = SBERT_MODEL.encode(
        ALL_TITLES,
        batch_size=64,
        show_progress_bar=True
    )

    print("✅ System ready")


# =========================
# HYBRID SIMILARITY
# =========================
def calculate_similarity(title1: str, title2: str) -> float:

    t1 = normalize_text(title1)
    t2 = normalize_text(title2)

    if not t1 or not t2:
        return 0.0

    # Semantic
    emb1 = SBERT_MODEL.encode([t1])
    emb2 = SBERT_MODEL.encode([t2])
    semantic_score = cosine_similarity(emb1, emb2)[0][0]

    # String
    tfidf = TfidfVectorizer().fit_transform([t1, t2])
    string_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

    # Phonetic
    meta1 = jellyfish.metaphone(t1)
    meta2 = jellyfish.metaphone(t2)
    phonetic_score = 1.0 if meta1 == meta2 and meta1 != "" else 0.0

    final_score = (
        0.2 * phonetic_score +
        0.3 * string_score +
        0.5 * semantic_score
    )

    return round(final_score * 100, 2)


# =========================
# RISK
# =========================
def classify_risk(score: float) -> str:
    if score >= 60:
        return "High"
    elif score >= 25:
        return "Medium"
    else:
        return "Low"


# =========================
# FAST COMPARE (MAIN)
# =========================
def compare_title(user_title: str, titles=None):

    try:
        if not user_title:
            return {"similarity": 0, "risk": "Low", "closest_title": None}

        if not ALL_TITLES or ALL_EMBEDDINGS is None:
            initialize_system()

        t1 = normalize_text(user_title)

        # Encode once
        input_emb = SBERT_MODEL.encode([t1])

        similarities = cosine_similarity(input_emb, ALL_EMBEDDINGS)[0]

        # 🔥 TOP 5 MATCHES
        top_indices = similarities.argsort()[-5:][::-1]

        top_matches = []
        scores_list = []

        for idx in top_indices:
            title = ALL_TITLES[idx]
            semantic_score = similarities[idx]

            # String score
            tfidf = TfidfVectorizer().fit_transform([t1, title])
            string_score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

            # Phonetic score
            meta1 = jellyfish.metaphone(t1)
            meta2 = jellyfish.metaphone(title)
            phonetic_score = 1.0 if meta1 == meta2 and meta1 != "" else 0.0

            # Hybrid score
            final_score = (
                0.2 * phonetic_score +
                0.3 * string_score +
                0.5 * semantic_score
            ) * 100

            scores_list.append(final_score)

            # 🔥 Explanation
            common_words = set(t1.split()) & set(title.split())

            top_matches.append({
                "title": title,
                "score": round(final_score, 2),
                "common_words": list(common_words)
            })

        # Best match
        best_match = top_matches[0]
        best_score = best_match["score"]

        # 🔥 Dynamic threshold
        avg_score = sum(scores_list) / len(scores_list)

        if best_score >= avg_score + 10:
            risk = "High"
        elif best_score >= avg_score:
            risk = "Medium"
        else:
            risk = "Low"

        # 🔥 Confidence
        confidence = min(100, round(best_score, 2))

        return {
            "similarity": best_score,
            "risk": risk,
            "closest_title": best_match["title"],

            # NEW FEATURES (won’t break UI)
            "top_matches": top_matches,
            "confidence": confidence,
            "dynamic_threshold": round(avg_score, 2)
        }

    except Exception as e:
        print("Error:", e)
        return {"similarity": 0, "risk": "Low", "closest_title": None}


# =========================
# COMPATIBILITY FUNCTIONS
# =========================
def initialize_prgi_system():
    initialize_system()


def find_closest_title_match(input_title: str):
    result = compare_title(input_title)

    return {
        "match": {"title": result["closest_title"]},
        "scores": {"hybrid": result["similarity"]},
        "rank_score": result["similarity"]
    }


def get_titles_from_database():
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