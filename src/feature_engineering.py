"""
Feature engineering for tabular models (Random Forest / XGBoost), following
the feature families named on slide 5 of the review deck:
  - daily/weekly cycle features
  - load variance
  - autocorrelation
  - change-point indicators
"""

import numpy as np
import pandas as pd


def _weekly_profile_features(series_row, dates):
    df = pd.DataFrame({"val": series_row.values, "dow": dates.dayofweek})
    by_dow = df.groupby("dow")["val"].mean()
    weekday_mean = by_dow.loc[0:4].mean()
    weekend_mean = by_dow.loc[5:6].mean()
    ratio = weekend_mean / (weekday_mean + 1e-6)
    return weekday_mean, weekend_mean, ratio


def _autocorrelation(x, lag):
    x = x - x.mean()
    n = len(x)
    if n <= lag:
        return 0.0
    num = np.sum(x[: n - lag] * x[lag:])
    den = np.sum(x**2) + 1e-9
    return num / den


def _change_point_score(x, window=14):
    """Largest jump in rolling mean between two adjacent windows, normalized."""
    if len(x) < 2 * window:
        return 0.0
    roll = pd.Series(x).rolling(window).mean().dropna().values
    if len(roll) < 2:
        return 0.0
    diffs = np.abs(np.diff(roll))
    return float(diffs.max() / (np.mean(x) + 1e-6))


def _capping_score(x):
    """How 'flat-topped' the distribution is: high percentile close to
    median indicates values are being suppressed toward a ceiling."""
    p95 = np.percentile(x, 95)
    p50 = np.percentile(x, 50)
    return float(p95 / (p50 + 1e-6))


def _zero_run_stats(x, thresh_ratio=0.1):
    mean_val = np.mean(x) + 1e-6
    near_zero = x < (thresh_ratio * mean_val)
    if not near_zero.any():
        return 0.0, 0
    # longest run of near-zero days
    runs, cur = [], 0
    for v in near_zero:
        cur = cur + 1 if v else 0
        runs.append(cur)
    return float(near_zero.mean()), int(max(runs))


def build_features(series: pd.DataFrame) -> pd.DataFrame:
    """series: consumers x dates (DatetimeIndex columns), cleaned kWh."""
    dates = series.columns
    feats = []
    for cons_id, row in series.iterrows():
        x = row.values.astype(float)

        mean_ = x.mean()
        std_ = x.std()
        cv = std_ / (mean_ + 1e-6)  # load variance, scale-free

        weekday_mean, weekend_mean, we_ratio = _weekly_profile_features(row, dates)

        ac1 = _autocorrelation(x, 1)
        ac7 = _autocorrelation(x, 7)
        ac30 = _autocorrelation(x, 30)

        cp_score = _change_point_score(x)
        cap_score = _capping_score(x)
        zero_frac, longest_zero_run = _zero_run_stats(x)

        # trend: slope of simple linear fit over the whole window
        t = np.arange(len(x))
        slope = np.polyfit(t, x, 1)[0] if len(x) > 1 else 0.0

        feats.append(
            {
                "CONS_NO": cons_id,
                "mean_kwh": mean_,
                "std_kwh": std_,
                "coeff_variation": cv,
                "weekday_mean": weekday_mean,
                "weekend_mean": weekend_mean,
                "weekend_weekday_ratio": we_ratio,
                "autocorr_lag1": ac1,
                "autocorr_lag7": ac7,
                "autocorr_lag30": ac30,
                "change_point_score": cp_score,
                "capping_score": cap_score,
                "zero_day_fraction": zero_frac,
                "longest_zero_run": longest_zero_run,
                "trend_slope": slope,
                "min_kwh": x.min(),
                "max_kwh": x.max(),
            }
        )

    feat_df = pd.DataFrame(feats).set_index("CONS_NO")
    return feat_df


if __name__ == "__main__":
    from data_preprocessing import load_and_clean

    meta, series = load_and_clean("data/smart_meter_data.csv")
    feats = build_features(series)
    print(feats.head())
    print(feats.shape)
