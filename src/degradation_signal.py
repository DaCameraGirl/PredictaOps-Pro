"""Is-it-degrading signal, kept separate from the RUL regression's point estimate.

"This bearing is degrading" and "this bearing has 46 snapshots left" are different
claims with different reliability. The first is a statistical deviation-from-healthy
check and holds up even when the exact RUL number is uncertain; the second depends
on the regression model generalizing correctly. Conflating them into one number
overstates how much the model actually knows.
"""
import numpy as np
import pandas as pd

from bearing_data import BEARING_COLS, FEATURE_NAMES

BASELINE_SNAPSHOTS = 100  # vibration features are flat over roughly this stretch at the start of the recording


class DegradationSignal:
    def __init__(self, table: pd.DataFrame):
        self._baselines = {}
        for bearing in BEARING_COLS:
            early = (
                table[table["bearing"] == bearing]
                .sort_values("timestamp")
                .iloc[:BASELINE_SNAPSHOTS]
            )
            mean = early[FEATURE_NAMES].mean()
            std = early[FEATURE_NAMES].std().replace(0, np.nan)
            self._baselines[bearing] = (mean, std)

    def evaluate(self, bearing: str, row: pd.Series) -> dict:
        mean, std = self._baselines[bearing]
        z = ((row[FEATURE_NAMES] - mean) / std).abs()
        deviation = float(z.mean(skipna=True))
        if deviation >= 4:
            confidence, is_degrading = "high", True
        elif deviation >= 2:
            confidence, is_degrading = "medium", True
        else:
            confidence, is_degrading = "low", False
        return {
            "is_degrading": is_degrading,
            "confidence": confidence,
            "deviation_z": round(deviation, 2),
        }
