"""
Synthetic smart-meter consumption data generator.

The project proposal (slide 5) references public SGCC-style smart meter
datasets with labeled normal/theft cases. Those datasets require manual
download from external portals (e.g. Kaggle) that this environment cannot
reach, so this script generates a synthetic dataset with the SAME shape and
statistical flavour: daily kWh readings per consumer over ~3 years, with a
subset of consumers exhibiting theft-like consumption patterns.

Swap this out for the real SGCC CSV by pointing `--input` in
src/data_preprocessing.py at your downloaded file — the rest of the
pipeline (feature engineering, models, risk scoring) does not need to
change, as long as the columns line up as CONS_NO, dates..., FLAG.
"""

import argparse
import numpy as np
import pandas as pd


def make_normal_series(n_days, rng, base_level):
    """Normal household: weekly seasonality + slow trend + noise."""
    t = np.arange(n_days)
    weekly = 1 + 0.15 * np.sin(2 * np.pi * t / 7)
    yearly = 1 + 0.25 * np.sin(2 * np.pi * t / 365 + rng.uniform(0, 2 * np.pi))
    trend = 1 + 0.05 * (t / n_days) * rng.uniform(-1, 1)
    noise = rng.normal(1, 0.08, n_days)
    series = base_level * weekly * yearly * trend * noise
    return np.clip(series, 0, None)


def apply_theft_pattern(series, rng, pattern=None):
    """
    Corrupt a normal series with one of several theft signatures:
    - 'partial_bypass': sustained percentage reduction in reported usage
    - 'periodic_zero': recurring days of near-zero reported usage
    - 'sudden_drop': a step-change drop partway through the series (meter
      tampering event) that persists afterward
    - 'capped': consumption is reported as if capped at a ceiling regardless
      of true usage (common tamper signature)
    """
    n = len(series)
    if pattern is None:
        pattern = rng.choice(
            ["partial_bypass", "periodic_zero", "sudden_drop", "capped"]
        )

    s = series.copy()
    if pattern == "partial_bypass":
        factor = rng.uniform(0.35, 0.65)
        s = s * factor
    elif pattern == "periodic_zero":
        period = rng.integers(5, 12)
        zero_days = np.arange(0, n, period)
        s[zero_days] *= rng.uniform(0.0, 0.15)
    elif pattern == "sudden_drop":
        change_point = rng.integers(int(n * 0.3), int(n * 0.8))
        factor = rng.uniform(0.3, 0.6)
        s[change_point:] *= factor
    elif pattern == "capped":
        cap = np.quantile(s, rng.uniform(0.25, 0.45))
        s = np.minimum(s, cap)

    return np.clip(s, 0, None), pattern


def generate(n_consumers=1200, n_days=365 * 2, theft_ratio=0.12, seed=42):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2022-01-01")
    dates = pd.date_range(start, periods=n_days, freq="D")

    n_theft = int(n_consumers * theft_ratio)
    is_theft = np.array([True] * n_theft + [False] * (n_consumers - n_theft))
    rng.shuffle(is_theft)

    rows = []
    labels = []
    patterns = []
    for i in range(n_consumers):
        base_level = rng.uniform(5, 40)  # kWh/day household scale
        series = make_normal_series(n_days, rng, base_level)
        pattern = "none"
        if is_theft[i]:
            series, pattern = apply_theft_pattern(series, rng)
        rows.append(series)
        labels.append(int(is_theft[i]))
        patterns.append(pattern)

    cons_ids = [f"C{i:05d}" for i in range(n_consumers)]
    df = pd.DataFrame(rows, index=cons_ids, columns=dates.strftime("%Y-%m-%d"))
    df.insert(0, "FLAG", labels)
    df.insert(1, "THEFT_PATTERN", patterns)
    df.index.name = "CONS_NO"
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_consumers", type=int, default=1200)
    parser.add_argument("--n_days", type=int, default=730)
    parser.add_argument("--theft_ratio", type=float, default=0.12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="data/smart_meter_data.csv")
    args = parser.parse_args()

    df = generate(args.n_consumers, args.n_days, args.theft_ratio, args.seed)
    df.to_csv(args.out)
    print(f"Wrote {df.shape[0]} consumers x {df.shape[1]-2} days -> {args.out}")
    print(f"Theft rate: {df['FLAG'].mean():.2%}")
