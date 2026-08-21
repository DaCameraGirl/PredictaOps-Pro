"""Health-state classification, kept separate from the RUL regression's point estimate.

"This bearing is degrading" and "this bearing has 46 snapshots left" are different
claims with different reliability. The first is a statistical deviation-from-healthy
check and holds up even when the exact RUL number is uncertain; the second depends
on the regression model generalizing correctly. Conflating them into one number
overstates how much the model actually knows.

States are driven by a transparent, documented rule, not a second trained model:
deviation (mean |z-score| across features) from each bearing's own early-life
baseline, smoothed with a rolling median so a single noisy snapshot can't flip the
state — only a sustained deviation can. This is a heuristic anomaly indicator, not a
diagnosed defect: it says "this looks statistically unlike how this bearing normally
runs," not "this bearing has fault X."
"""
import numpy as np
import pandas as pd

from bearing_data import BEARING_COLS, FEATURE_NAMES

BASELINE_SNAPSHOTS = 100  # vibration features are flat over roughly this stretch at the start of the recording
ROLLING_WINDOW = 5  # snapshots a deviation must persist across before it counts as a state change, not a spike

# Upper bound (exclusive) of each state's mean-|z| band.
STATE_THRESHOLDS = [
    ("healthy", 2.0),
    ("watch", 4.0),
    ("warning", 8.0),
    ("critical", float("inf")),
]


def _classify(z: float) -> str:
    if np.isnan(z):
        return "insufficient_evidence"
    for name, upper in STATE_THRESHOLDS:
        if z < upper:
            return name
    return "critical"  # unreachable given the inf sentinel above, kept for clarity


class DegradationSignal:
    def __init__(self, table: pd.DataFrame):
        self._series: dict[str, pd.DataFrame] = {}
        self._first_detected: dict[str, str | None] = {}

        for bearing in BEARING_COLS:
            sub = table[table["bearing"] == bearing].sort_values("timestamp").reset_index(drop=True)
            baseline = sub.iloc[:BASELINE_SNAPSHOTS]
            mean = baseline[FEATURE_NAMES].mean()
            std = baseline[FEATURE_NAMES].std().replace(0, np.nan)

            z_scores = (sub[FEATURE_NAMES] - mean) / std
            deviation = z_scores.abs().mean(axis=1, skipna=True)
            smoothed = deviation.rolling(ROLLING_WINDOW, min_periods=1).median()

            # The baseline window itself hasn't got an independent baseline to compare
            # against, so it's not evidence either way rather than "healthy by definition."
            state = smoothed.apply(_classify)
            state.iloc[:BASELINE_SNAPSHOTS] = "insufficient_evidence"

            is_spike = (deviation > 2 * smoothed) & (smoothed < STATE_THRESHOLDS[0][1])

            frame = pd.DataFrame({
                "timestamp": sub["timestamp"],
                "deviation_z": deviation,
                "smoothed_z": smoothed,
                "state": state,
                "is_spike": is_spike,
            })
            top_feature = z_scores.abs().idxmax(axis=1)
            frame["top_feature"] = top_feature
            self._series[bearing] = frame.set_index("timestamp")

            persistent_states = {"watch", "warning", "critical"}
            first = frame[frame["state"].isin(persistent_states)]
            self._first_detected[bearing] = (
                first.iloc[0]["timestamp"].isoformat() if not first.empty else None
            )

    def timeline(self, bearing: str) -> dict:
        frame = self._series[bearing]
        events = {"baseline_established": None, "first_persistent_anomaly": None,
                  "warning_threshold_crossed": None, "critical_threshold_crossed": None}
        if len(frame) > BASELINE_SNAPSHOTS:
            events["baseline_established"] = frame.index[BASELINE_SNAPSHOTS - 1].isoformat()
        events["first_persistent_anomaly"] = self._first_detected[bearing]
        for state, key in [("warning", "warning_threshold_crossed"), ("critical", "critical_threshold_crossed")]:
            hit = frame[frame["state"] == state]
            if not hit.empty:
                events[key] = hit.index[0].isoformat()
        return events

    def evaluate(self, bearing: str, ts) -> dict:
        row = self._series[bearing].loc[ts]
        state = row["state"]
        return {
            "health_state": state,
            "is_degrading": state in ("watch", "warning", "critical"),
            "confidence": {
                "insufficient_evidence": "none",
                "healthy": "high",
                "watch": "medium",
                "warning": "high",
                "critical": "high",
            }[state],
            "deviation_z": round(float(row["deviation_z"]), 2) if not np.isnan(row["deviation_z"]) else None,
            "smoothed_z": round(float(row["smoothed_z"]), 2) if not np.isnan(row["smoothed_z"]) else None,
            "is_spike": bool(row["is_spike"]),
            "top_feature": row["top_feature"],
            "first_detected": self._first_detected[bearing],
        }
