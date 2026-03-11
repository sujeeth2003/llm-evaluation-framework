import pandas as pd
from evaluation.metrics import compute_accuracy
from evaluation.hallucination_detection import hallucination_rate
import matplotlib.pyplot as plt

df = pd.read_csv("results/benchmark_results.csv")

models = df["model"].unique()

summary = []

for m in models:

    sub = df[df["model"] == m]

    acc = compute_accuracy(sub)
    halluc = hallucination_rate(sub)

    summary.append({
        "model": m,
        "accuracy": acc,
        "hallucination": halluc
    })

summary_df = pd.DataFrame(summary)

print(summary_df)

summary_df.to_csv("results/model_summary.csv", index=False)

plt.bar(summary_df["model"], summary_df["accuracy"])
plt.title("Model Accuracy Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("results/accuracy_plot.png")
