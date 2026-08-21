"""Automated profiling for the extracted bearing vibration features."""
import pandas as pd

from bearing_data import BEARING_COLS, FAILED_BEARING, FAILURE_MODE, FEATURE_NAMES


def profile_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in FEATURE_NAMES:
        series = df[col]
        rows.append(
            {
                "feature": col,
                "missing": int(series.isna().sum()),
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "std": series.std(),
            }
        )
    return pd.DataFrame(rows).set_index("feature")


def degradation_trends(df: pd.DataFrame) -> pd.Series:
    """Correlation of each feature with elapsed time, for the bearing that actually failed."""
    failed = df[df["bearing"] == FAILED_BEARING].sort_values("timestamp")
    elapsed = (failed["timestamp"] - failed["timestamp"].iloc[0]).dt.total_seconds()
    correlations = {feat: elapsed.corr(failed[feat]) for feat in FEATURE_NAMES}
    return pd.Series(correlations, name="corr_with_elapsed_time").sort_values(
        key=abs, ascending=False
    )


def feature_trend(df: pd.DataFrame, bearing: str, max_points: int = 200) -> list[dict]:
    """Downsampled per-feature time series for one bearing's whole observed life."""
    sub = df[df["bearing"] == bearing].sort_values("timestamp")
    step = max(1, len(sub) // max_points)
    sampled = sub.iloc[::step]
    return [
        {"timestamp": row["timestamp"].isoformat(), **{f: float(row[f]) for f in FEATURE_NAMES}}
        for _, row in sampled.iterrows()
    ]


def summarize(df: pd.DataFrame) -> dict:
    trends = degradation_trends(df)
    span = df["timestamp"].max() - df["timestamp"].min()
    return {
        "n_snapshots": df["timestamp"].nunique(),
        "n_bearings": len(BEARING_COLS),
        "recording_span_days": round(span.total_seconds() / 86400, 1),
        "sampling_rate_hz": 20000,
        "failed_bearing": FAILED_BEARING,
        "failure_mode": FAILURE_MODE,
        "top_degradation_features": trends.head(4).round(3).to_dict(),
    }


if __name__ == "__main__":
    from bearing_data import build_feature_table

    table = build_feature_table()
    summary = summarize(table)
    print(summary)
