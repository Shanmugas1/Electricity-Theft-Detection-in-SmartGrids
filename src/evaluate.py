"""
Evaluation utilities. Slide 6 names accuracy, precision, recall, F1 as the
primary metrics, with emphasis on high precision at top-ranked alerts —
so this module reports both standard classification metrics and
precision@k / recall@k for the risk-ranked inspection list.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def classification_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba) if len(set(y_true)) > 1 else float("nan"),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def precision_recall_at_k(y_true, y_proba, k_values=(0.05, 0.10, 0.20)):
    """Precision/recall among the top-k fraction of consumers by risk
    score — this is the operational metric: 'of the inspections we can
    actually afford, how many are real theft cases'."""
    n = len(y_true)
    order = np.argsort(-y_proba)
    y_true_sorted = np.asarray(y_true)[order]
    total_positives = y_true_sorted.sum()

    rows = []
    for frac in k_values:
        k = max(1, int(np.ceil(frac * n)))
        top_k = y_true_sorted[:k]
        precision_at_k = top_k.sum() / k
        recall_at_k = top_k.sum() / total_positives if total_positives > 0 else float("nan")
        rows.append(
            {"top_fraction": frac, "k": k, "precision@k": precision_at_k, "recall@k": recall_at_k}
        )
    return pd.DataFrame(rows)


def summarize_models(results: dict, y_test) -> pd.DataFrame:
    """results: {model_name: y_proba}. Returns a comparison table."""
    rows = []
    for name, proba in results.items():
        m = classification_metrics(y_test, proba)
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
            }
        )
    return pd.DataFrame(rows).set_index("model").round(4)
