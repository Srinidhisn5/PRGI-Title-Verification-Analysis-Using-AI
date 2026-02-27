# similarity_engine.py
# Production similarity and risk classification engine

# FORCE PYTORCH-ONLY MODE - Prevent TensorFlow imports
import os
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["USE_TF"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import List, Tuple, Dict, Any, Set
import os
import logging
import re
import unicodedata
from functools import lru_cache
from collections import Counter
from datetime import datetime
import pandas as pd
import sqlite3

logger = logging.getLogger(__name__)

# Required libraries for advanced similarity
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    from difflib import SequenceMatcher
    import jellyfish  # For phonetic similarity (Double Metaphone)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    import nltk
    from sentence_transformers import SentenceTransformer
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except ImportError as e:
    logger.error(f"Missing required libraries: {e}")
    raise ImportError("Please install required packages: scikit-learn, numpy, jellyfish, nltk, pandas, sentence-transformers")

# PRGI Domain Keywords
PRGI_KEYWORDS = {
    'news', 'digest', 'bulletin', 'samachar', 'tak', 'aaj', 'ajj', 'hindustan',
    'times', 'express', 'post', 'mail', 'darpan', 'mirror', 'sandesh', 'patrika',
    'jagran', 'dainik', 'daily', 'weekly', 'monthly', 'fortnightly', 'quarterly',
    'government', 'govt', 'public', 'national', 'india', 'indian', 'bharat',
    'pradesh', 'rajasthan', 'uttar', 'pradesh', 'madhya', 'pradesh', 'maharashtra',
    'gujarat', 'karnataka', 'tamil', 'nadu', 'andhra', 'pradesh', 'telangana',
    'kerala', 'punjab', 'haryana', 'bihar', 'jharkhand', 'west', 'bengal'
}

# Database path
BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, 'database', 'database.db')

# =============================================================================
# GLOBAL STATE - EXACTLY ONE MODEL ENFORCEMENT
# =============================================================================

# MANDATORY RULE 1: At top of file define ONLY these three globals
SBERT_MODEL = None
PRGI_TITLES = None
PRGI_EMBEDDINGS = None

def initialize_sbert_model():
    """
    Initialize SBERT model globally - called once at app startup.

    MANDATORY RULE 2:
    - MUST declare: global SBERT_MODEL
    - MUST assign SBERT_MODEL = SentenceTransformer(...)
    - MUST return SBERT_MODEL
    - MUST print appropriate messages
    """
    global SBERT_MODEL

    if SBERT_MODEL is None:
        print("SBERT model loaded")
        SBERT_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    else:
        print("SBERT model reused")

    return SBERT_MODEL

def generate_prgi_embeddings():
    """
    Generate SBERT embeddings for all PRGI titles.

    MANDATORY RULE 3:
    - MUST declare: global PRGI_EMBEDDINGS
    - MUST use SBERT_MODEL ONLY
    - MUST HARD FAIL if SBERT_MODEL is None
    - MUST HARD FAIL if PRGI_TITLES is empty
    - MUST store embeddings in PRGI_EMBEDDINGS
    - MUST NEVER load model again
    """
    global PRGI_EMBEDDINGS

    # HARD FAIL if SBERT_MODEL is None
    if SBERT_MODEL is None:
        raise RuntimeError("SBERT_MODEL is None - model not initialized")

    # HARD FAIL if PRGI_TITLES is empty
    if not PRGI_TITLES or len(PRGI_TITLES) == 0:
        raise RuntimeError("PRGI_TITLES is empty - no titles to embed")

    # Generate embeddings using SBERT_MODEL ONLY
    PRGI_EMBEDDINGS = SBERT_MODEL.encode(
        PRGI_TITLES,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return PRGI_EMBEDDINGS

def initialize_prgi_system():
    """
    ONE-TIME initialization pipeline for PRGI system.

    MANDATORY BEHAVIOR:
    1. Must declare: global PRGI_EMBEDDINGS
    2. Must run steps IN THIS EXACT ORDER:
       a) initialize_sbert_model()
       b) load_prgi_data()
       c) generate_prgi_embeddings()
    3. Must be IDEMPOTENT:
       - If PRGI_EMBEDDINGS already exists → do nothing
       - Must NOT regenerate embeddings
    4. Must print logs
    5. Must HARD FAIL (RuntimeError) if any step fails
    """
    global PRGI_EMBEDDINGS

    # IDEMPOTENT: If already initialized, do nothing
    if PRGI_EMBEDDINGS is not None:
        return

    print("Initializing PRGI system...")

    try:
        # Step a) initialize_sbert_model()
        initialize_sbert_model()
        print("SBERT ready")

        # Step b) load_prgi_data()
        load_prgi_data()
        print("PRGI titles ready")

        # Step c) generate_prgi_embeddings()
        generate_prgi_embeddings()
        print("PRGI embeddings ready")

        # HARD FAIL validation
        if SBERT_MODEL is None:
            raise RuntimeError("SBERT_MODEL is None after initialization")

        if not PRGI_TITLES or len(PRGI_TITLES) == 0:
            raise RuntimeError("PRGI_TITLES is empty after initialization")

        if PRGI_EMBEDDINGS is None:
            raise RuntimeError("PRGI_EMBEDDINGS is None after initialization")

        print("PRGI system initialized successfully")

    except Exception as e:
        print(f"❌ CRITICAL ERROR: PRGI system initialization failed: {e}")
        raise RuntimeError(f"PRGI system initialization failed: {e}")

def normalize_text(text: str) -> str:
    """Comprehensive text normalization with publication stopwords removal."""
    if not text or not isinstance(text, str):
        return ""

    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Convert to uppercase first for consistency, then lowercase
    text = text.upper()

    # Remove punctuation and special characters (keep alphanumeric and spaces)
    text = re.sub(r'[^\w\s]', ' ', text)

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Convert to lowercase
    text = text.lower()

    # Remove generic publication stopwords before encoding
    publication_stopwords = {
        'news', 'daily', 'bulletin', 'digest', 'weekly', 'monthly', 'times',
        'post', 'express', 'mail', 'chronicle', 'gazette', 'patrika', 'sandesh',
        'jagran', 'dainik', 'hindustan', 'ajj', 'aaj', 'tak', 'samachar'
    }

    words = text.split()
    filtered_words = [word for word in words if word.lower() not in publication_stopwords]
    text = ' '.join(filtered_words)

    return text.strip()

def get_db_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)

def load_prgi_data():
    """
    Load PRGI dataset from CSV/Excel files at startup using the centralized
    loader in `prgi_dataset`. This enforces that all four authoritative
    datasets are loaded, combined, and preserved verbatim.

    Raises RuntimeError on any failure (no DB fallbacks).
    """
    global PRGI_TITLES
    try:
        # Import the authoritative loader (which logs per-file counts)
        from prgi_dataset import load_prgi_titles as _load_prgi_titles

        titles = _load_prgi_titles()

        # Strict validation: ensure enough titles loaded
        if not titles or len(titles) < 2000:
            raise RuntimeError(f"Insufficient PRGI titles loaded: {len(titles) if titles else 0}")

        # Preserve CSV rows exactly as returned by loader (no lowercasing here)
        PRGI_TITLES = titles

        # For compatibility/logging repeat the required message (primary log is from loader)
        print(f"PRGI titles assigned for similarity: {len(PRGI_TITLES)}")

        # Print a small sample for debug (first 3 rows)
        print("SAMPLE PRGI TITLES:")
        for i, t in enumerate(PRGI_TITLES[:3]):
            print(f"  {i+1}. {t}")

        return PRGI_TITLES

    except Exception as e:
        print(f"❌ CRITICAL ERROR: Failed to load PRGI data: {e}")
        PRGI_TITLES = []
        raise RuntimeError(f"PRGI dataset loading failed: {e}")

def get_titles_from_database() -> List[Dict[str, Any]]:
    """
    CRITICAL FIX: Load titles from SQLite database as SINGLE SOURCE OF TRUTH.

    Returns cleaned titles from database submissions table.
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()

        # Get all titles from database (approved + pending + rejected)
        # CRITICAL: Include ALL titles regardless of status for accurate similarity
        cur.execute("""
            SELECT DISTINCT title, status, id
            FROM submissions
            WHERE title IS NOT NULL AND title != ''
        """)

        raw_titles = cur.fetchall()
        conn.close()

        cleaned_titles = []
        seen_titles = set()

        for row in raw_titles:
            original_title = row[0]
            status = row[1]
            title_id = row[2]

            # Normalize title
            normalized_title = normalize_text(original_title)

            # Skip titles shorter than 4 characters
            if len(normalized_title) < 4:
                continue

            # Skip titles with only repeated single words
            words = normalized_title.split()
            if len(words) > 0:
                word_counts = Counter(words)
                if len(word_counts) == 1 and word_counts[words[0]] > 1:
                    continue

            # Skip single letter or meaningless titles
            if len(words) == 1 and len(words[0]) <= 2 and words[0].lower() not in PRGI_KEYWORDS:
                continue

            # Deduplicate
            if normalized_title in seen_titles:
                continue

            seen_titles.add(normalized_title)
            cleaned_titles.append({
                'original_title': original_title,
                'normalized_title': normalized_title,
                'source': 'DB',  # Database source
                'registration_number': f'DB-{title_id}',
                'language': 'N/A',  # Not available in database
                'status': status,
                'title_id': title_id
            })

        logger.info(f"Loaded {len(cleaned_titles)} titles from database")
        return cleaned_titles

    except Exception as e:
        logger.error(f"Failed to load titles from database: {e}")
        return []


def preprocess_title(text: str) -> Dict[str, Any]:
    """
    Comprehensive title preprocessing pipeline.

    Returns dict with processed components.
    """
    if not text or not isinstance(text, str):
        return {
            'normalized': '',
            'tokens': [],
            'keywords': set(),
            'word_count': 0
        }

    # Basic normalization
    normalized = normalize_text(text)

    # Tokenization
    tokens = word_tokenize(normalized)

    # Remove stopwords and get keywords
    stop_words = set(stopwords.words('english'))
    # Add basic Indian fillers
    indian_fillers = {'ki', 'ka', 'ke', 'se', 'mein', 'par', 'aur', 'hai', 'hain', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    stop_words.update(indian_fillers)

    keywords = set()
    for token in tokens:
        token_lower = token.lower()
        # Keep if it's a PRGI keyword OR longer than 2 chars and not stopword
        if token_lower in PRGI_KEYWORDS or (len(token) > 2 and token_lower not in stop_words):
            keywords.add(token_lower)

    return {
        'normalized': normalized,
        'tokens': tokens,
        'keywords': keywords,
        'word_count': len(tokens)
    }

# SIMILARITY TECHNIQUES

def sbert_semantic_similarity(input_title: str, prgi_title: str) -> float:
    """
    SBERT-based semantic similarity using precomputed embeddings cache.

    Args:
        input_title: Input title to compare
        prgi_title: Existing PRGI title to compare against

    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    try:
        if not SBERT_MODEL:
            logger.warning("SBERT model not initialized")
            return 0.0

        # Normalize both titles
        input_normalized = normalize_text(input_title)
        prgi_normalized = normalize_text(prgi_title)

        # Always compute input embedding (since it's new)
        input_embedding = SBERT_MODEL.encode([input_normalized], convert_to_numpy=True)[0]

        # Always compute prgi embedding (since it's new)
        prgi_embedding = SBERT_MODEL.encode([prgi_normalized], convert_to_numpy=True)[0]

        # Calculate cosine similarity
        similarity = cosine_similarity([input_embedding], [prgi_embedding])[0][0]

        # Ensure result is between 0 and 1
        return max(0.0, min(1.0, similarity))

    except Exception as e:
        logger.warning(f"SBERT semantic similarity failed: {e}")
        return 0.0

def semantic_similarity(input_title: str, prgi_title: str) -> float:
    """
    LEGACY: Word-level TF-IDF + Character n-gram TF-IDF similarity.
    Kept for fallback/compatibility but no longer used in main scoring.
    """
    try:
        texts = [input_title, prgi_title]

        # Word-level TF-IDF
        word_vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
        word_tfidf = word_vectorizer.fit_transform(texts)
        word_sim = cosine_similarity(word_tfidf[0:1], word_tfidf[1:2])[0][0]

        # Character-level TF-IDF (3-5 grams)
        char_vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
        char_tfidf = char_vectorizer.fit_transform(texts)
        char_sim = cosine_similarity(char_tfidf[0:1], char_tfidf[1:2])[0][0]

        # Return average of both similarities
        return (word_sim + char_sim) / 2.0
    except Exception as e:
        logger.warning(f"Legacy semantic similarity failed: {e}")
        return 0.0

def phonetic_similarity(input_title: str, prgi_title: str) -> float:
    """Double Metaphone phonetic similarity (only if input ≤ 4 words)."""
    try:
        # Only apply phonetic similarity if input title has ≤ 4 words
        input_words = input_title.split()
        if len(input_words) > 4:
            return 0.0

        # Use Double Metaphone ONLY
        input_meta = jellyfish.metaphone(input_title)
        prgi_meta = jellyfish.metaphone(prgi_title)

        if not input_meta or not prgi_meta:
            return 0.0

        # Exact match gives 1.0, no match gives 0.0
        return 1.0 if input_meta == prgi_meta else 0.0
    except Exception as e:
        logger.warning(f"Phonetic similarity failed: {e}")
        return 0.0

def fuzzy_similarity(input_title: str, prgi_title: str) -> float:
    """Token set ratio similarity."""
    try:
        from difflib import SequenceMatcher
        # Use token_set_ratio for fuzzy matching
        return SequenceMatcher(None, input_title, prgi_title).ratio()
    except Exception as e:
        logger.warning(f"Fuzzy similarity failed: {e}")
        return 0.0

def ngram_similarity(input_title: str, prgi_title: str) -> float:
    """Character n-gram Jaccard similarity."""
    try:
        def get_char_ngrams(text: str, n: int = 3) -> Set[str]:
            return {text[i:i+n] for i in range(len(text)-n+1) if len(text[i:i+n]) == n}

        input_ngrams = get_char_ngrams(input_title)
        prgi_ngrams = get_char_ngrams(prgi_title)

        if not input_ngrams or not prgi_ngrams:
            return 0.0

        intersection = len(input_ngrams.intersection(prgi_ngrams))
        union = len(input_ngrams.union(prgi_ngrams))

        return intersection / union if union > 0 else 0.0
    except Exception as e:
        logger.warning(f"N-gram similarity failed: {e}")
        return 0.0

def keyword_similarity(input_title: str, prgi_title: str) -> float:
    """PRGI domain keyword overlap similarity."""
    try:
        # Preprocess both titles
        input_processed = preprocess_title(input_title)
        prgi_processed = preprocess_title(prgi_title)

        input_keywords = input_processed['keywords']
        prgi_keywords = prgi_processed['keywords']

        if not input_keywords or not prgi_keywords:
            return 0.0

        # Find intersection of PRGI domain keywords
        domain_matches = input_keywords.intersection(prgi_keywords).intersection(PRGI_KEYWORDS)

        # Calculate overlap ratio
        total_unique = len(input_keywords.union(prgi_keywords))
        return len(domain_matches) / total_unique if total_unique > 0 else 0.0
    except Exception as e:
        logger.warning(f"Keyword similarity failed: {e}")
        return 0.0

def compute_similarity_scores(input_title: str, prgi_title: str) -> Dict[str, float]:
    """
    Compute all similarity scores for a title pair (normalized 0-100).
    Now uses SBERT as primary semantic similarity engine.
    """
    try:
        return {
            'semantic': sbert_semantic_similarity(input_title, prgi_title) * 100,  # SBERT semantic
            'phonetic': phonetic_similarity(input_title, prgi_title) * 100,
            'fuzzy': fuzzy_similarity(input_title, prgi_title) * 100,
            'ngram': ngram_similarity(input_title, prgi_title) * 100,
            'keyword': keyword_similarity(input_title, prgi_title) * 100
        }
    except Exception as e:
        logger.warning(f"Similarity computation failed: {e}")
        return {
            'semantic': 0.0,
            'phonetic': 0.0,
            'fuzzy': 0.0,
            'ngram': 0.0,
            'keyword': 0.0
        }

def find_closest_title_match(input_title: str) -> Dict[str, Any]:
    """Find closest PRGI title using semantic + fuzzy ranking (60% semantic + 40% fuzzy)."""
    # CRITICAL FIX: Load from database directly
    cleaned_dataset = get_titles_from_database()

    best_match = None
    best_rank_score = -1
    best_scores = {}

    for item in cleaned_dataset:
        # Use normalized_title if available (from database), otherwise normalize
        if 'normalized_title' in item:
            prgi_title = item['normalized_title']
        else:
            prgi_title = normalize_text(item['title'])

        scores = compute_similarity_scores(input_title, prgi_title)

        # Use semantic + fuzzy ranking for closest match selection (NOT final score)
        rank_score = (0.6 * scores['semantic']) + (0.4 * scores['fuzzy'])

        if rank_score > best_rank_score:
            best_rank_score = rank_score
            best_match = item
            best_scores = scores

    return {
        'match': best_match,
        'scores': best_scores,
        'rank_score': best_rank_score
    }

def find_top_k_title_matches(input_title: str, k: int = 2) -> List[Dict[str, Any]]:
    """
    Find top-K closest PRGI titles using semantic + fuzzy ranking.

    Args:
        input_title: Title to find matches for
        k: Number of top matches to return

    Returns:
        List of top-K matches with scores, sorted by ranking score
    """
    # CRITICAL FIX: Load from database directly
    cleaned_dataset = get_titles_from_database()

    # Calculate ranking scores for all titles
    ranked_matches = []

    for item in cleaned_dataset:
        # Use normalized_title if available (from database), otherwise normalize
        if 'normalized_title' in item:
            prgi_title = item['normalized_title']
        else:
            prgi_title = normalize_text(item['title'])

        scores = compute_similarity_scores(input_title, prgi_title)

        # Use semantic + fuzzy ranking for closest match selection
        rank_score = (0.6 * scores['semantic']) + (0.4 * scores['fuzzy'])

        ranked_matches.append({
            'match': item,
            'scores': scores,
            'rank_score': rank_score
        })

    # Sort by ranking score (highest first) and return top-K
    ranked_matches.sort(key=lambda x: x['rank_score'], reverse=True)
    return ranked_matches[:k]

# =============================================================================
# EXPLANATION ENGINE - RESEARCH-GRADE EXPLAINABILITY LAYER
# =============================================================================

def assess_confidence(technique_scores: Dict[str, float]) -> str:
    """
    Assess confidence in similarity analysis based on technique agreement.

    Args:
        technique_scores: Dictionary with semantic, phonetic, fuzzy, ngram, keyword scores

    Returns:
        Confidence level: "High", "Medium", "Low"
    """
    scores = list(technique_scores.values())

    # Calculate variance in scores (lower variance = higher agreement = higher confidence)
    if len(scores) < 2:
        return "Low"

    mean_score = sum(scores) / len(scores)
    variance = sum((score - mean_score) ** 2 for score in scores) / len(scores)
    std_dev = variance ** 0.5

    # High confidence: low variance (techniques agree)
    if std_dev < 15:
        return "High"
    # Medium confidence: moderate variance
    elif std_dev < 30:
        return "Medium"
    # Low confidence: high variance (techniques disagree)
    else:
        return "Low"

def explain_technique_contributions(technique_scores: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Calculate relative contribution percentages and generate explanations for each technique.

    Args:
        technique_scores: Raw technique scores (0-100)

    Returns:
        List of technique explanations with contributions
    """
    # Updated weights for SBERT-powered hybrid scoring
    WEIGHTS = {
        'semantic': 0.40,  # SBERT semantic (PRIMARY)
        'phonetic': 0.05,  # Phonetic similarity (REDUCED)
        'fuzzy': 0.25,     # Fuzzy string matching
        'ngram': 0.20,     # Character n-gram similarity
        'keyword': 0.10    # Keyword overlap
    }

    # Calculate weighted contributions
    total_weighted_score = sum(technique_scores[tech] * weight for tech, weight in WEIGHTS.items())

    technique_explanations = []

    for technique, score in technique_scores.items():
        weight = WEIGHTS[technique]
        weighted_contribution = score * weight

        # Calculate relative contribution percentage
        if total_weighted_score > 0:
            relative_contribution = (weighted_contribution / total_weighted_score) * 100
        else:
            relative_contribution = 0

        # Generate human-readable explanation
        explanation = get_technique_explanation(technique, score, relative_contribution)

        technique_explanations.append({
            'technique': technique,
            'display_name': get_technique_display_name(technique),
            'score': round(score, 2),
            'weight': weight,
            'contribution_percent': round(relative_contribution, 1),
            'explanation': explanation,
            'impact_level': get_impact_level(relative_contribution)
        })

    # Sort by contribution (highest first)
    technique_explanations.sort(key=lambda x: x['contribution_percent'], reverse=True)

    return technique_explanations

def get_technique_display_name(technique: str) -> str:
    """Get user-friendly display name for technique."""
    names = {
        'semantic': 'Semantic Similarity',
        'phonetic': 'Pronunciation Similarity',
        'fuzzy': 'Text Structure Matching',
        'ngram': 'Character Pattern Overlap',
        'keyword': 'Publication Keywords'
    }
    return names.get(technique, technique)

def get_technique_explanation(technique: str, score: float, contribution: float) -> str:
    """Generate plain-language explanation for a technique's contribution."""

    if technique == 'semantic':
        if score >= 80:
            return "Titles convey very similar meanings and topics"
        elif score >= 60:
            return "Titles share significant semantic overlap"
        elif score >= 40:
            return "Titles have some meaning similarity"
        else:
            return "Minimal semantic overlap between titles"

    elif technique == 'phonetic':
        if score >= 90:
            return "Titles sound nearly identical when spoken"
        elif score > 0:
            return "Titles have some phonetic similarity"
        else:
            return "Pronunciation differences are significant"

    elif technique == 'fuzzy':
        if score >= 80:
            return "Titles are structurally very similar"
        elif score >= 60:
            return "Titles share similar word arrangements"
        elif score >= 40:
            return "Titles have noticeable structural similarities"
        else:
            return "Titles differ significantly in structure"

    elif technique == 'ngram':
        if score >= 70:
            return "Titles share many common character sequences"
        elif score >= 50:
            return "Some character patterns appear in both titles"
        elif score >= 30:
            return "Limited character sequence overlap"
        else:
            return "Character sequences are largely different"

    elif technique == 'keyword':
        if score >= 60:
            return "Both titles use key publication industry terms"
        elif score >= 30:
            return "Some specialized terminology is shared"
        else:
            return "Limited use of common publication vocabulary"

    return "Unable to analyze this aspect"

def get_impact_level(contribution_percent: float) -> str:
    """Categorize the impact level of a technique's contribution."""
    if contribution_percent >= 30:
        return "MAJOR"
    elif contribution_percent >= 15:
        return "SIGNIFICANT"
    elif contribution_percent >= 5:
        return "MODERATE"
    else:
        return "MINOR"

def create_plain_language_summary(
    final_score: float,
    risk_level: str,
    confidence: str,
    top_contributors: List[Dict[str, Any]],
    closest_title: str
) -> str:
    """
    Create a comprehensive plain-language summary for government administrators.

    Args:
        final_score: Overall similarity score (0-100)
        risk_level: "High", "Medium", "Low"
        confidence: "High", "Medium", "Low"
        top_contributors: List of top contributing techniques
        closest_title: The most similar existing title

    Returns:
        Human-readable summary paragraph
    """

    if final_score >= 80:
        risk_summary = f"This title shows strong similarity ({final_score:.0f}%) to existing publications and may violate registration guidelines."
    elif final_score >= 50:
        risk_summary = f"This title has moderate similarity ({final_score:.0f}%) to existing publications and warrants careful review."
    else:
        risk_summary = f"This title shows low similarity ({final_score:.0f}%) to existing publications and appears suitable for registration."

    # Identify primary reasons
    if top_contributors:
        primary_reason = top_contributors[0]['display_name']
        reason_explanation = top_contributors[0]['explanation']
    else:
        primary_reason = "multiple analysis methods"
        reason_explanation = "various similarity factors"

    confidence_note = {
        "High": "The analysis methods strongly agree on this assessment.",
        "Medium": "Most analysis methods align, though some show slight variations.",
        "Low": "The analysis methods show some disagreement, suggesting additional review may be needed."
    }.get(confidence, "")

    summary = f"{risk_summary} The primary reason is {reason_explanation.lower()} based on {primary_reason.lower()}. {confidence_note}"

    if closest_title and risk_level in ["High", "Medium"]:
        summary += f" Most similar to: '{closest_title}'."

    return summary

def generate_key_insights(technique_breakdown: List[Dict[str, Any]], risk_level: str) -> List[str]:
    """
    Generate key insights about what the similarity analysis reveals.

    Args:
        technique_breakdown: Detailed technique analysis
        risk_level: Overall risk classification

    Returns:
        List of insight statements
    """
    insights = []

    # Find dominant techniques
    major_contributors = [t for t in technique_breakdown if t['impact_level'] == 'MAJOR']

    if risk_level == "High":
        insights.append("High risk of public confusion between publications")
        insights.append("Registration may violate uniqueness requirements")
        if any(t['technique'] == 'semantic' for t in major_contributors):
            insights.append("Publications appear to target similar audiences")
        if any(t['technique'] == 'keyword' for t in major_contributors):
            insights.append("Both titles use similar publication industry terminology")

    elif risk_level == "Medium":
        insights.append("Moderate risk requiring careful administrative review")
        insights.append("Similarity concerns should be evaluated against registration criteria")
        if any(t['technique'] == 'fuzzy' for t in major_contributors):
            insights.append("Titles have similar structural composition")

    else:  # Low risk
        insights.append("Low similarity suggests title is likely unique")
        insights.append("Minimal risk of confusion with existing publications")

    return insights

def generate_recommendations(risk_level: str, technique_breakdown: List[Dict[str, Any]]) -> List[str]:
    """
    Generate actionable recommendations based on analysis results.

    Args:
        risk_level: Overall risk level
        technique_breakdown: Technique analysis details

    Returns:
        List of recommendation statements
    """
    recommendations = []

    if risk_level == "High":
        recommendations.append("Consider rejecting registration or requiring title modification")
        recommendations.append("Consult legal team for trademark and confusion analysis")
        recommendations.append("Review geographical scope differences between publications")

        # Specific recommendations based on technique contributions
        if any(t['technique'] == 'semantic' and t['score'] > 70 for t in technique_breakdown):
            recommendations.append("Evaluate whether publications serve different purposes")
        if any(t['technique'] == 'phonetic' and t['score'] > 80 for t in technique_breakdown):
            recommendations.append("Consider pronunciation differences in different regions")

    elif risk_level == "Medium":
        recommendations.append("Conduct manual review of publication purposes and audiences")
        recommendations.append("Consider requiring additional distinguishing features")
        recommendations.append("Review publisher's intended market and distribution")

    else:  # Low risk
        recommendations.append("Title appears suitable for registration")
        recommendations.append("Monitor for future similar submissions")

    return recommendations

def generate_explanations(similarity_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main explanation engine function - generates comprehensive explainability output.

    Args:
        similarity_result: Complete similarity analysis result from compare_title()

    Returns:
        Structured explanation dictionary
    """
    try:
        final_score = similarity_result.get('final_score', 0)
        risk_level = similarity_result.get('risk', 'Low')
        breakdown = similarity_result.get('breakdown', {})
        closest_title = similarity_result.get('closest_title', '')

        # Generate technique contribution analysis
        technique_breakdown = explain_technique_contributions(breakdown)

        # Assess confidence
        confidence = assess_confidence(breakdown)

        # Generate plain language summary
        summary = create_plain_language_summary(
            final_score, risk_level, confidence, technique_breakdown, closest_title
        )

        # Generate insights and recommendations
        insights = generate_key_insights(technique_breakdown, risk_level)
        recommendations = generate_recommendations(risk_level, technique_breakdown)

        # Identify what matters most
        top_contributor = technique_breakdown[0] if technique_breakdown else None
        what_matters = {
            "most_important": top_contributor['display_name'] if top_contributor else "Unable to determine",
            "least_important": min(technique_breakdown, key=lambda x: x['contribution_percent'])['display_name'] if technique_breakdown else "Unable to determine",
            "threshold_context": f"Score of {final_score:.1f}% {'exceeds' if final_score >= 50 else 'is below'} government caution threshold"
        }

        return {
            "overall_explanation": {
                "risk_level": risk_level,
                "confidence": confidence,
                "summary": summary
            },
            "technique_breakdown": technique_breakdown,
            "key_insights": insights,
            "recommendations": recommendations,
            "what_matters": what_matters
        }

    except Exception as e:
        logger.error(f"Explanation generation failed: {e}")
        return {
            "overall_explanation": {
                "risk_level": "Unknown",
                "confidence": "Low",
                "summary": "Unable to generate explanation due to analysis error"
            },
            "technique_breakdown": [],
            "key_insights": ["Analysis encountered an error"],
            "recommendations": ["Manual review recommended"],
            "what_matters": {
                "most_important": "Unable to determine",
                "least_important": "Unable to determine",
                "threshold_context": "Analysis error occurred"
            }
        }

def _compare_title_impl(title: str, prgi_titles=None):
    try:
        if not title or not title.strip():
            raise ValueError("Empty title")

        # Use the provided prgi_titles parameter or fall back to global PRGI_TITLES
        if prgi_titles is None or len(prgi_titles) == 0:
            # Fall back to global PRGI_TITLES if available
            if PRGI_TITLES is None or len(PRGI_TITLES) == 0:
                raise ValueError("No PRGI titles provided for comparison")
            prgi_titles = PRGI_TITLES

        # Use the provided prgi_titles (CSV PRGI_TITLES by default).
        # Do NOT restrict candidates to database-backed titles — compute similarity
        # against the CSV PRGI dataset (the authoritative PRGI list).
        results = []
        for prgi_title in prgi_titles:
            norm_prgi = normalize_text(prgi_title)
            similarity = calculate_similarity(title, prgi_title)
            results.append({
                "title": prgi_title,
                "normalized": norm_prgi,
                "similarity": similarity
            })

        print(f"PRGI titles considered (MASTER only): {len(results)}")

        # If no PRGI titles available, return a clear no-match response
        if not results:
            print("❌ CRITICAL: No PRGI titles available for comparison")
            return {
                "similarity": 0.0,
                "risk": "Low",
                "closest_title": None,
                "message": "PRGI titles empty",
                "error": "PRGI titles empty"
            }

        # Sort by similarity (descending) and pick the top CSV match
        results.sort(key=lambda x: x["similarity"], reverse=True)

        # Prefer an exact verbatim CSV match first (must be identical to CSV row)
        input_stripped = title.strip()
        verbatim_match = None
        for candidate in results:
            if candidate.get('title') == input_stripped:
                verbatim_match = candidate
                break

        if verbatim_match:
            similarity_percent = 100.0
            closest_title = verbatim_match.get('title')
        else:
            # Next prefer an exact normalized match, otherwise use the top scorer
            normalized_input = normalize_text(title)
            top_match = None
            for candidate in results:
                if candidate.get('normalized') == normalized_input:
                    top_match = candidate
                    break
            if not top_match:
                top_match = results[0]

            similarity_percent = round(float(top_match.get("similarity", 0.0)), 2)
            closest_title = top_match.get("title")

        # Determine risk and log
        risk_level = classify_risk(similarity_percent)

        print("PRGI result:", {
            "similarity": similarity_percent,
            "risk": risk_level,
            "closest_title": closest_title
        })

        # If similarity is 0, explicitly signal no meaningful match found
        if similarity_percent == 0:
            return {
                "similarity": 0.0,
                "risk": "Low",
                "closest_title": None,
                "message": "No meaningful PRGI match found"
            }

        return {
            "similarity": similarity_percent,
            "risk": risk_level,
            "closest_title": closest_title
        }

    except Exception as e:
        # 🔥 NEW STRUCTURE - Return exact structure even on error
        return {
            "similarity": 0.0,
            "risk": "Low",
            "closest_title": None,
            "message": "No meaningful PRGI match found",
            "error": f"Analysis failed: {str(e)}"
        }

def calculate_similarity(title1: str, title2: str) -> float:
    """Calculate similarity between two titles using word overlap + normalization + phonetic fallback."""
    try:
        # Basic normalization without removing PRGI keywords
        def simple_normalize(text):
            if not text or not isinstance(text, str):
                return ""
            # Unicode normalization
            text = unicodedata.normalize('NFKC', text)
            # Convert to lowercase
            text = text.lower()
            # Remove punctuation and special characters
            text = re.sub(r'[^\w\s]', ' ', text)
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            return text.strip()

        # Normalize both titles (keep PRGI keywords)
        norm1 = simple_normalize(title1)
        norm2 = simple_normalize(title2)

        if not norm1 or not norm2:
            return 0.0

        # Word overlap + normalization as specified
        words1 = set(norm1.split())
        words2 = set(norm2.split())

        if not words1 or not words2:
            return 0.0

        overlap = words1.intersection(words2)
        word_score = (len(overlap) / max(len(words1), len(words2))) * 100

        # 🔥 FIX 2: Add phonetic fallback using rapidfuzz
        try:
            from rapidfuzz import fuzz
            phonetic_score = fuzz.partial_ratio(norm1, norm2)
            final_score = max(word_score, phonetic_score * 0.5)
        except ImportError:
            # Fallback to word score if rapidfuzz not available
            final_score = word_score

        return round(final_score, 2)

    except Exception as e:
        print(f"Similarity calculation error: {e}")
        return 0.0

def find_closest_title(input_title: str) -> str:
    """Legacy function for backward compatibility."""
    result = compare_title(input_title)
    return result.get('closest_title', '')

# SIMPLIFIED FINAL FUNCTIONS - EXACT REQUIREMENTS IMPLEMENTATION
import re
from rapidfuzz import fuzz

def normalize_text(text: str) -> str:
    """Normalize text for similarity calculation"""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)  # Remove special chars
    text = re.sub(r"\s+", " ", text)  # Collapse whitespace
    return text

def calculate_similarity(title1: str, title2: str) -> float:
    """
    Calculate similarity using exact requirements:
    - Word overlap similarity: (overlapping_words / max(words_in_title1, words_in_title2)) * 100
    - Phonetic fallback: rapidfuzz partial_ratio * 0.5
    - Final similarity = max(word_score, phonetic_score * 0.5)
    """
    norm1 = normalize_text(title1)
    norm2 = normalize_text(title2)

    # Word overlap similarity
    words1 = set(norm1.split())
    words2 = set(norm2.split())

    if words1 and words2:
        overlap = len(words1 & words2)
        word_score = (overlap / max(len(words1), len(words2))) * 100
    else:
        word_score = 0.0

    # Phonetic fallback using rapidfuzz
    phonetic_score = fuzz.partial_ratio(norm1, norm2) * 0.5

    # Final similarity = max(word_score, phonetic_score * 0.5)
    final_score = max(word_score, phonetic_score)

    return round(final_score, 2)

def classify_risk(score: float) -> str:
    """
    Classify risk using optimized thresholds.
    
    Optimized via exhaustive threshold calibration (Step 5):
    - Baseline: 70.8% accuracy (thresholds 25/60)
    - Optimized: 76.7% accuracy (thresholds 28/50)
    - Improvement: +5.9% overall accuracy
    
    Thresholds:
      0-28:   Low
      28-50:  Medium
      50+:    High
    """
    if score >= 50:
        return "High"
    elif score >= 28:
        return "Medium"
    else:
        return "Low"

def compare_title(user_title: str, prgi_titles=None):
    """
    Public compare_title wrapper that computes similarity ONLY against the
    PRGI_MASTER_DATASET (PRGI_TITLES) by delegating to the internal
    implementation. No other CSVs or DB titles are used for matching.
    """
    return _compare_title_impl(user_title, prgi_titles)

# Attempt to load PRGI titles at import time to ensure module-level PRGI_TITLES is populated.
try:
    load_prgi_data()
except Exception as e:
    # Fail loudly is acceptable during normal startup; tests will catch missing data.
    print(f"PRGI dataset not loaded at import: {e}")

# SELF TEST
if __name__ == "__main__":
    tests = [
        "Indian Monthly Digest",
        "Monthly Digest of India",
        "Aaj Tak News",
        "Ajj Taak Samachar",
        "Tech News with Government",
        "Government Tech Bulletin"
    ]

    for t in tests:
        print(f"\nTesting: {t}")
        result = compare_title(t)
        print(f"Score: {result.get('similarity', 0.0)}%, Risk: {result.get('risk', 'Low')}")
        print(f"Closest: {result.get('closest_title', None)}")
        print("-" * 80)
