import numpy as np
import pandas as pd

from bearing_data import BEARING_COLS, FEATURE_NAMES
from degradation_signal import DegradationSignal


def _synthetic_table(n=200, jump_at=150, jump_size=10.0):
    """One bearing, flat healthy baseline, then a sustained jump in every feature
    starting at `jump_at` — a case with a known correct classification."""
    rng = np.random.default_rng(1)
    timestamps = pd.date_range("2004-01-01", periods=n, freq="10min")
    rows = []
    for bearing in BEARING_COLS:
        for i in range(n):
            base = {f: rng.normal(loc=0, scale=0.01) for f in FEATURE_NAMES}
            if bearing == BEARING_COLS[0] and i >= jump_at:
                base = {f: v + jump_size for f, v in base.items()}
            rows.append({"bearing": bearing, "timestamp": timestamps[i], **base})
    return pd.DataFrame(rows)


def test_baseline_window_is_insufficient_evidence_not_healthy_by_default():
    table = _synthetic_table()
    sig = DegradationSignal(table)
    early_ts = table[table.bearing == BEARING_COLS[0]].sort_values("timestamp")["timestamp"].iloc[0]
    result = sig.evaluate(BEARING_COLS[0], early_ts)
    assert result["health_state"] == "insufficient_evidence"
    assert result["confidence"] == "none"


def test_sustained_deviation_is_classified_as_degrading():
    table = _synthetic_table(n=200, jump_at=150)
    sig = DegradationSignal(table)
    bearing = BEARING_COLS[0]
    late_ts = table[table.bearing == bearing].sort_values("timestamp")["timestamp"].iloc[-1]
    result = sig.evaluate(bearing, late_ts)
    assert result["is_degrading"] is True
    assert result["health_state"] in ("watch", "warning", "critical")


def test_untouched_bearing_stays_healthy():
    table = _synthetic_table(n=200, jump_at=150)
    sig = DegradationSignal(table)
    bearing = BEARING_COLS[1]  # never perturbed
    late_ts = table[table.bearing == bearing].sort_values("timestamp")["timestamp"].iloc[-1]
    result = sig.evaluate(bearing, late_ts)
    assert result["health_state"] == "healthy"
    assert result["is_degrading"] is False


def test_single_snapshot_spike_does_not_immediately_flip_state():
    """A single anomalous reading, surrounded by otherwise-healthy readings, should
    be visible as a spike but not immediately promote the smoothed state — that's
    the entire point of the rolling-median smoothing."""
    n = 200
    rng = np.random.default_rng(2)
    timestamps = pd.date_range("2004-01-01", periods=n, freq="10min")
    rows = []
    spike_index = 150
    for i in range(n):
        base = {f: rng.normal(loc=0, scale=0.01) for f in FEATURE_NAMES}
        if i == spike_index:
            base = {f: v + 20.0 for f, v in base.items()}
        rows.append({"bearing": BEARING_COLS[0], "timestamp": timestamps[i], **base})
    for bearing in BEARING_COLS[1:]:
        for i in range(n):
            rows.append({"bearing": bearing, "timestamp": timestamps[i],
                         **{f: rng.normal(loc=0, scale=0.01) for f in FEATURE_NAMES}})
    table = pd.DataFrame(rows)

    sig = DegradationSignal(table)
    spike_ts = timestamps[spike_index]
    result = sig.evaluate(BEARING_COLS[0], spike_ts)
    assert result["health_state"] == "healthy", "one spike alone should not cross into watch/warning/critical"

    next_ts = timestamps[spike_index + 1]
    next_result = sig.evaluate(BEARING_COLS[0], next_ts)
    assert next_result["health_state"] == "healthy"


def test_first_detected_matches_first_persistent_watch_or_above():
    table = _synthetic_table(n=200, jump_at=150)
    sig = DegradationSignal(table)
    timeline = sig.timeline(BEARING_COLS[0])
    assert timeline["first_persistent_anomaly"] is not None
    # first detection should be at/after the jump, and well before the end of the series
    jump_ts = pd.Timestamp("2004-01-01") + pd.Timedelta(minutes=10 * 150)
    assert pd.Timestamp(timeline["first_persistent_anomaly"]) >= jump_ts


def test_never_failed_bearing_has_no_critical_threshold_event_by_construction():
    table = _synthetic_table(n=200, jump_at=150)
    sig = DegradationSignal(table)
    timeline = sig.timeline(BEARING_COLS[1])  # never perturbed
    assert timeline["warning_threshold_crossed"] is None
    assert timeline["critical_threshold_crossed"] is None
