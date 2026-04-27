import pandas as pd
from similarity_engine import compare_title, initialize_system
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
initialize_system() 
df = pd.read_csv("database/test_dataset.csv")

y_true = []
y_pred = []

print("\n🔍 Running Evaluation...\n")

for _, row in df.iterrows():
    title = row["input_title"]
    language = row["language"]
    expected = row["expected"]

    result = compare_title(title, language=language)
    predicted = result["risk"]

    y_true.append(expected)
    y_pred.append(predicted)

    print(f"{title} → Expected: {expected}, Predicted: {predicted}, Score: {result['similarity']:.2f}")

# 📊 METRICS
print("\n📊 FINAL RESULTS\n")

accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:\n")
print(classification_report(y_true, y_pred))

print("Confusion Matrix:\n")
print(confusion_matrix(y_true, y_pred))