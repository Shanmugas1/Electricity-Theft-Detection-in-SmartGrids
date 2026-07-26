"""
Generate comparison charts from the outputs/ produced by run_pipeline.py:
- bar chart of accuracy/precision/recall/F1 per model
- precision@k / recall@k curve for the inspection alert list
"""

import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def plot_model_comparison(out_dir=OUT_DIR):
    df = pd.read_csv(os.path.join(out_dir, "model_comparison.csv"), index_col="model")
    metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    ax = df[metrics].plot(kind="bar", figsize=(9, 5), rot=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — Energy Theft Detection")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved", path)


def plot_precision_recall_at_k(out_dir=OUT_DIR):
    df = pd.read_csv(os.path.join(out_dir, "precision_recall_at_k.csv"))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(df["top_fraction"] * 100, df["precision@k"], marker="o", label="Precision@k")
    ax.plot(df["top_fraction"] * 100, df["recall@k"], marker="o", label="Recall@k")
    ax.set_xlabel("Top % of consumers flagged")
    ax.set_ylabel("Score")
    ax.set_title("Precision/Recall at top-k inspection alerts")
    ax.set_ylim(0, 1.05)
    ax.legend()
    plt.tight_layout()
    path = os.path.join(out_dir, "precision_recall_at_k.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print("Saved", path)


if __name__ == "__main__":
    plot_model_comparison()
    plot_precision_recall_at_k()
