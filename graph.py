import matplotlib.pyplot as plt

methods = ["Phonetic", "String", "Semantic", "Hybrid"]
f1_scores = [0.59, 0.64, 0.72, 0.76]

plt.figure()
plt.bar(methods, f1_scores)

plt.xlabel("Similarity Methods")
plt.ylabel("F1 Score")
plt.title("Performance Comparison of Similarity Techniques")

plt.ylim(0.5, 0.8)

plt.show()
for i, v in enumerate(f1_scores):
    plt.text(i, v + 0.005, str(v), ha='center')