from similarity_engine import normalize_text, PRGI_TITLES, SBERT_MODEL
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import jellyfish
import random

# ------------------------
# ENSURE DATA IS LOADED
# ------------------------
if not PRGI_TITLES:
    from similarity_engine import load_prgi_data
    load_prgi_data()

# ------------------------
# PHONETIC FUNCTION
# ------------------------
def phonetic_similarity(text1, text2):
    words1 = text1.split()
    words2 = text2.split()

    matches = 0
    total = min(len(words1), len(words2))

    for w1, w2 in zip(words1, words2):
        if jellyfish.metaphone(w1) == jellyfish.metaphone(w2):
            matches += 1

    return matches / total if total > 0 else 0


# ------------------------
# CREATE TEST DATA (REALISTIC)
# ------------------------
test_cases = []

# ------------------------
# POSITIVE CASES (NOISY VARIATIONS)
# ------------------------
for i in range(50):
    t = PRGI_TITLES[i]
    words = t.split()

    # remove random word
    if len(words) > 3:
        words.pop(random.randint(0, len(words) - 1))

    # introduce small typo
    if len(words) > 0:
        words[0] = words[0][:max(1, len(words[0]) - 1)]

    modified = " ".join(words)

    test_cases.append((t, modified, 1))


# ------------------------
# NEGATIVE CASES (HARD NEGATIVES)
# ------------------------
for i in range(50):
    t1 = PRGI_TITLES[i]
    t2 = PRGI_TITLES[i + 400]

    t1_words = t1.split()
    t2_words = t2.split()

    # mix words to confuse model
    mixed = " ".join(t2_words[:2] + t1_words[-2:])

    test_cases.append((t1, mixed, 0))


# ------------------------
# STORAGE
# ------------------------
y_true = []
y_pred_phonetic = []
y_pred_string = []
y_pred_semantic = []
y_pred_hybrid = []


# ------------------------
# EVALUATION LOOP
# ------------------------
for t1, t2, label in test_cases:

    t1n = normalize_text(t1)
    t2n = normalize_text(t2)

    # ------------------
    # PHONETIC
    # ------------------
    phonetic_sim = phonetic_similarity(t1n, t2n)
    phonetic_score = 1 if phonetic_sim >= 0.8 else 0

    # ------------------
    # STRING (TF-IDF)
    # ------------------
    tfidf = TfidfVectorizer().fit_transform([t1n, t2n])
    string_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    string_score = 1 if string_sim >= 0.3 else 0

    # ------------------
    # SEMANTIC (SBERT)
    # ------------------
    emb1 = SBERT_MODEL.encode([t1n])
    emb2 = SBERT_MODEL.encode([t2n])
    semantic_sim = cosine_similarity(emb1, emb2)[0][0]
    semantic_score = 1 if semantic_sim >= 0.8 else 0

    # ------------------
    # HYBRID
    # ------------------
    hybrid_sim = (
        0.2 * phonetic_score +
        0.3 * string_sim +
        0.5 * semantic_sim
    )
    hybrid_score = 1 if hybrid_sim >= 0.55 else 0

    # ------------------
    # STORE RESULTS
    # ------------------
    y_true.append(label)
    y_pred_phonetic.append(phonetic_score)
    y_pred_string.append(string_score)
    y_pred_semantic.append(semantic_score)
    y_pred_hybrid.append(hybrid_score)


# ------------------------
# FINAL RESULTS
# ------------------------
print("\n===== FINAL F1 SCORES =====")
print("Phonetic :", round(f1_score(y_true, y_pred_phonetic), 2))
print("String   :", round(f1_score(y_true, y_pred_string), 2))
print("Semantic :", round(f1_score(y_true, y_pred_semantic), 2))
print("Hybrid   :", round(f1_score(y_true, y_pred_hybrid), 2))