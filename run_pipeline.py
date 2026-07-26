"""
End-to-end pipeline: preprocess -> feature engineer -> train RF/XGBoost/LSTM
-> evaluate -> risk-rank -> write outputs/

Mirrors the pipeline described on slide 5 of the review deck:
  collect/preprocess -> feature engineering -> model training ->
  score & rank -> inspection alerts
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_preprocessing import load_and_clean
from feature_engineering import build_features
from models import train_random_forest, train_xgboost, train_lstm
from evaluate import summarize_models, precision_recall_at_k, classification_metrics
from risk_scoring import combine_scores, build_alert_list


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "smart_meter_data.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def main(data_path=DATA_PATH, out_dir=OUT_DIR, test_size=0.25, random_state=42):
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("STEP 1/5  Loading and cleaning smart meter data")
    print("=" * 70)
    meta, series = load_and_clean(data_path)
    print(f"  {series.shape[0]} consumers, {series.shape[1]} days, "
          f"theft rate {meta['FLAG'].mean():.2%}")

    print("\n" + "=" * 70)
    print("STEP 2/5  Feature engineering")
    print("=" * 70)
    feats = build_features(series)
    feats = feats.join(meta["FLAG"])
    print(f"  {feats.shape[1] - 1} engineered features")

    cons_ids = feats.index.values
    X = feats.drop(columns=["FLAG"]).values
    y = feats["FLAG"].values
    seq = series.loc[feats.index].values  # aligned raw sequences for LSTM

    idx = np.arange(len(cons_ids))
    idx_train, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=random_state
    )

    X_train, X_test = X[idx_train], X[idx_test]
    y_train, y_test = y[idx_train], y[idx_test]
    seq_train, seq_test = seq[idx_train], seq[idx_test]
    cons_test = cons_ids[idx_test]

    print("\n" + "=" * 70)
    print("STEP 3/5  Training models")
    print("=" * 70)

    print("  Random Forest ...")
    rf_model, rf_proba = train_random_forest(X_train, y_train, X_test, random_state)

    print("  XGBoost ...")
    xgb_model, xgb_proba = train_xgboost(X_train, y_train, X_test, random_state)

    print("  LSTM (this can take a minute on CPU) ...")
    lstm_model, lstm_proba = train_lstm(seq_train, y_train, seq_test, random_state=random_state)

    print("\n" + "=" * 70)
    print("STEP 4/5  Evaluation")
    print("=" * 70)
    results = {"random_forest": rf_proba, "xgboost": xgb_proba, "lstm": lstm_proba}
    summary = summarize_models(results, y_test)
    print(summary)
    summary.to_csv(os.path.join(out_dir, "model_comparison.csv"))

    best_name = summary["f1"].idxmax()
    print(f"\n  Best model by F1: {best_name}")

    pr_at_k = precision_recall_at_k(y_test, results[best_name])
    print("\n  Precision/Recall at top-k (best model):")
    print(pr_at_k.to_string(index=False))
    pr_at_k.to_csv(os.path.join(out_dir, "precision_recall_at_k.csv"), index=False)

    print("\n" + "=" * 70)
    print("STEP 5/5  Risk scoring & inspection alert list")
    print("=" * 70)
    combined = combine_scores(
        {"random_forest": rf_proba, "xgboost": xgb_proba, "lstm": lstm_proba},
        weights={"random_forest": 0.35, "xgboost": 0.4, "lstm": 0.25},
    )
    alerts = build_alert_list(cons_test, combined, y_true=y_test)
    alerts.to_csv(os.path.join(out_dir, "inspection_alert_list.csv"), index=False)
    print(f"  Top 10 highest-risk consumers:")
    print(alerts.head(10).to_string(index=False))

    metrics_json = {name: classification_metrics(y_test, p) for name, p in results.items()}
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    print(f"\nAll outputs written to: {out_dir}/")
    return summary, alerts


if __name__ == "__main__":
    main()
