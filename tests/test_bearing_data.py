from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from bearing_data import (
    BEARING_COLS,
    FAILED_BEARING,
    DatasetValidationError,
    _snapshot_features,
    add_rul,
    validate_raw_dataset,
)


def test_snapshot_features_on_known_sine_wave():
    """A pure sine wave has textbook-known RMS and crest factor, independent of this
    codebase — a real ground truth to check the feature math against, not just
    "does it run"."""
    t = np.linspace(0, 1, 20480, endpoint=False)
    amplitude = 2.0
    signal = amplitude * np.sin(2 * np.pi * 50 * t)

    feats = _snapshot_features(signal)

    assert feats["mean"] == pytest.approx(0.0, abs=1e-6)
    assert feats["rms"] == pytest.approx(amplitude / np.sqrt(2), rel=1e-3)
    assert feats["peak_to_peak"] == pytest.approx(2 * amplitude, rel=1e-3)
    assert feats["crest_factor"] == pytest.approx(np.sqrt(2), rel=1e-2)


def test_snapshot_features_handles_constant_signal_without_dividing_by_zero():
    signal = np.zeros(20480)
    feats = _snapshot_features(signal)
    assert feats["kurtosis"] == 0.0
    assert feats["skew"] == 0.0
    assert feats["crest_factor"] == 0.0


def test_add_rul_counts_down_to_exactly_zero_at_last_recorded_snapshot():
    rows = []
    start = datetime(2004, 1, 1)
    for bearing in BEARING_COLS:
        for i in range(10):
            rows.append({"bearing": bearing, "timestamp": start + timedelta(minutes=10 * i)})
    df = pd.DataFrame(rows)

    labeled = add_rul(df)

    assert set(labeled["bearing"].unique()) == {FAILED_BEARING}
    assert labeled.sort_values("timestamp")["RUL"].iloc[-1] == 0
    assert labeled.sort_values("timestamp")["RUL"].iloc[0] == 9
    assert (labeled["RUL"] >= 0).all()


def test_add_rul_only_labels_the_bearing_with_a_known_failure():
    """Bearings 2-4 never failed in this test — right-censored data must never get a
    fabricated RUL label."""
    rows = [{"bearing": b, "timestamp": datetime(2004, 1, 1)} for b in BEARING_COLS]
    labeled = add_rul(pd.DataFrame(rows))
    assert "bearing_2" not in labeled["bearing"].values
    assert "bearing_3" not in labeled["bearing"].values
    assert "bearing_4" not in labeled["bearing"].values


def test_validate_raw_dataset_rejects_wrong_snapshot_count(tmp_path):
    (tmp_path / "2004.02.12.10.32.39").write_text("0 0 0 0\n")
    with pytest.raises(DatasetValidationError, match=r"expected .* snapshots"):
        validate_raw_dataset(tmp_path)


def test_validate_raw_dataset_rejects_unparseable_filename(tmp_path):
    (tmp_path / "not-a-timestamp.txt").write_text("0 0 0 0\n")
    with pytest.raises(DatasetValidationError, match="unparseable"):
        validate_raw_dataset(tmp_path)


def test_validate_raw_dataset_rejects_malformed_snapshot_shape(tmp_path, monkeypatch):
    import bearing_data as bd

    monkeypatch.setattr(bd, "EXPECTED_N_SNAPSHOTS", 3)
    monkeypatch.setattr(bd, "EXPECTED_SAMPLES_PER_SNAPSHOT", 4)  # keep files tiny for the test

    for i in range(3):
        ts = datetime(2004, 1, 1) + timedelta(minutes=10 * i)
        content = "\n".join("0 0 0 0" for _ in range(4))
        if i == 1:
            content = "0 0 0"  # wrong shape: too few rows and columns
        (tmp_path / ts.strftime("%Y.%m.%d.%H.%M.%S")).write_text(content)

    with pytest.raises(DatasetValidationError, match="malformed"):
        bd.validate_raw_dataset(tmp_path)


def test_feature_table_has_no_missing_values(feature_table):
    from bearing_data import FEATURE_NAMES

    assert not feature_table[FEATURE_NAMES].isna().any().any()


def test_feature_table_has_no_duplicate_bearing_timestamp_pairs(feature_table):
    dupes = feature_table.duplicated(subset=["bearing", "timestamp"])
    assert not dupes.any()
