"""
Load raw smart-meter consumption data and clean it:
- parse wide (CONS_NO x date columns) format
- handle missing readings (linear interpolation, then fill remaining edges)
- clip negative/implausible spikes
- return a tidy long-format frame plus the original wide matrix for
  sequence models
"""

import numpy as np
import pandas as pd


META_COLS = ["FLAG", "THEFT_PATTERN"]


def load_raw(path):
    df = pd.read_csv(path, index_col="CONS_NO")
    return df


def split_meta_and_series(df):
    meta_cols = [c for c in META_COLS if c in df.columns]
    meta = df[meta_cols].copy()
    series = df.drop(columns=meta_cols)
    series.columns = pd.to_datetime(series.columns)
    return meta, series


def clean_series(series, max_reasonable_kwh=500):
    """Interpolate missing values and clip implausible spikes/negatives."""
    s = series.copy()
    s[s < 0] = np.nan
    s[s > max_reasonable_kwh] = np.nan

    # interpolate along the time axis (columns), per consumer (row)
    s = s.interpolate(axis=1, limit_direction="both")

    # any consumer still fully NaN (interpolation had nothing to work with)
    # gets filled with 0 rather than dropped, so shapes stay consistent
    s = s.fillna(0.0)
    return s


def load_and_clean(path):
    raw = load_raw(path)
    meta, series = split_meta_and_series(raw)
    clean = clean_series(series)
    return meta, clean


if __name__ == "__main__":
    meta, series = load_and_clean("data/smart_meter_data.csv")
    print("Consumers:", series.shape[0], "Days:", series.shape[1])
    print("Missing after cleaning:", series.isna().sum().sum())
    print(meta["FLAG"].value_counts())
