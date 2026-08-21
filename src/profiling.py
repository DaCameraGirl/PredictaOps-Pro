"""Automated dataset profiling for the C-MAPSS sensor data."""
import pandas as pd

from data_loader import SENSOR_COLS, SETTING_COLS


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column stats: missingness, range, and whether it's degenerate (no signal)."""
    cols = SETTING_COLS + SENSOR_COLS
    rows = []
    for col in cols:
        series = df[col]
        rows.append(
            {
                "column": col,
                "missing": int(series.isna().sum()),
                "min": series.min(),
                "max": series.max(),
                "mean": series.mean(),
                "std": series.std(),
                "constant": bool(series.std() < 1e-8),
            }
        )
    return pd.DataFrame(rows).set_index("column")


def profile_units(df: pd.DataFrame) -> pd.DataFrame:
    """Per-unit lifetime (cycle count) so we can see fleet variability."""
    lifetimes = df.groupby("unit")["cycle"].max().rename("lifetime_cycles")
    return lifetimes.to_frame()


def degradation_trends(df: pd.DataFrame) -> pd.Series:
    """Correlation of each sensor with cycle number, averaged across units.

    A value near 0 means the sensor doesn't move as the engine degrades;
    a value near +-1 means it tracks degradation closely and is a strong
    signal for the model.
    """
    correlations = {}
    for sensor in SENSOR_COLS:
        per_unit_corr = df.groupby("unit").apply(
            lambda g: g["cycle"].corr(g[sensor]), include_groups=False
        )
        correlations[sensor] = per_unit_corr.mean()
    return pd.Series(correlations, name="mean_corr_with_cycle").sort_values(
        key=abs, ascending=False
    )


def summarize(df: pd.DataFrame) -> dict:
    col_profile = profile_columns(df)
    unit_profile = profile_units(df)
    trends = degradation_trends(df)
    return {
        "n_rows": len(df),
        "n_units": df["unit"].nunique(),
        "constant_sensors": col_profile[col_profile["constant"]].index.tolist(),
        "lifetime_cycles": {
            "min": int(unit_profile["lifetime_cycles"].min()),
            "max": int(unit_profile["lifetime_cycles"].max()),
            "mean": float(unit_profile["lifetime_cycles"].mean()),
        },
        "top_degradation_sensors": trends.head(5).round(3).to_dict(),
        "column_profile": col_profile,
        "unit_profile": unit_profile,
    }


if __name__ == "__main__":
    from data_loader import load_train

    train = load_train()
    summary = summarize(train)
    print(f"rows={summary['n_rows']} units={summary['n_units']}")
    print("constant sensors (no signal, safe to drop):", summary["constant_sensors"])
    print("lifetime cycles:", summary["lifetime_cycles"])
    print("top sensors correlated with degradation:", summary["top_degradation_sensors"])
