"""
Combine anomaly/classifier probabilities into a single risk score per
consumer and produce a ranked inspection alert list (slide 5: "Risk
Scoring & Alerts" — combine anomaly score and classifier probability to
produce a ranked inspection list).
"""

import pandas as pd


def combine_scores(proba_dict: dict, weights: dict = None) -> pd.Series:
    """
    proba_dict: {"random_forest": array, "xgboost": array, "lstm": array}
                all aligned to the same consumer order.
    weights:    optional per-model weight; defaults to equal weighting.
    """
    df = pd.DataFrame(proba_dict)
    if weights is None:
        weights = {k: 1.0 for k in proba_dict}
    w = pd.Series(weights)
    w = w / w.sum()
    combined = (df * w).sum(axis=1)
    return combined


def build_alert_list(cons_ids, combined_scores, y_true=None, top_n=None):
    df = pd.DataFrame({"CONS_NO": cons_ids, "risk_score": combined_scores})
    if y_true is not None:
        df["actual_theft"] = y_true
    df = df.sort_values("risk_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", df.index + 1)
    if top_n:
        df = df.head(top_n)
    return df
