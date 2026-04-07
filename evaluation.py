from similarity_engine import compare_title
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# =========================
# TEST DATASET (50+ cases recommended)
# =========================
test_cases = [
    # SIMILAR
    ("indian news daily", "similar"),
    ("india news journal", "similar"),
    ("laxmi times", "similar"),
    ("sandhya halchal", "similar"),
    ("bankura barta", "similar"),
    ("daily india news", "similar"),
    ("news india daily", "similar"),
    ("bharat news", "similar"),

    # UNIQUE
    ("random unique xyz", "unique"),
    ("abcd qwerty zzzz", "unique"),
    ("unknown publication name", "unique"),
    ("zzzz random title", "unique"),
    ("completely new brand name", "unique"),
    ("no match publication", "unique"),
]

# =========================
# EVALUATION
# =========================
def evaluate():

    y_true = []
    y_pred = []

    print("\n🔍 Running Full Evaluation...\n")

    for text, expected in test_cases:

        result = compare_title(text)
        score = result.get("similarity", 0)

        # 🔥 Threshold (tuned)
        if score >= 45:
            predicted = "similar"
        else:
            predicted = "unique"

        y_true.append(1 if expected == "similar" else 0)
        y_pred.append(1 if predicted == "similar" else 0)

        print(f"{text} → Score: {score:.2f} | Pred: {predicted} | Exp: {expected}")

    # =========================
    # METRICS
    # =========================
    accuracy = sum([1 for i in range(len(y_true)) if y_true[i] == y_pred[i]]) / len(y_true)

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)

    print("\n==============================")
    print(f"✅ Accuracy: {accuracy*100:.2f}%")
    print(f"✅ Precision: {precision*100:.2f}%")
    print(f"✅ Recall: {recall*100:.2f}%")
    print(f"✅ F1 Score: {f1*100:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("==============================\n")

    # =========================
    # GRAPH
    # =========================
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    values = [accuracy*100, precision*100, recall*100, f1*100]

    plt.figure()
    plt.bar(labels, values)
    plt.title("Model Performance Metrics")
    plt.xlabel("Metrics")
    plt.ylabel("Percentage")
    plt.show()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    evaluate()