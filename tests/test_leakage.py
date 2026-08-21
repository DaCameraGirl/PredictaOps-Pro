"""Automated guards against future-data leakage in the RUL validation pipeline.

These are the tests the walk-forward backtest exists to satisfy. If any of these
start failing, the validation numbers reported to the app and the submission are
no longer trustworthy — treat a failure here as blocking, not cosmetic.
"""
import numpy as np
import pandas as pd
import pytest

from bearing_data import FEATURE_NAMES, add_rul
from train_bearing import walk_forward_backtest


@pytest.fixture(scope="module")
def labeled(feature_table):
    labeled = add_rul(feature_table)
    labeled["RUL"] = labeled["RUL"].clip(upper=400)
    return labeled


def test_every_fold_trains_strictly_before_it_tests(labeled):
    _, _, _, fold_meta = walk_forward_backtest(labeled)
    assert len(fold_meta) > 0
    for fold in fold_meta:
        assert fold["train_end"] < fold["test_start"], (
            f"fold {fold['fold']} trains up to {fold['train_end']} but tests from "
            f"{fold['test_start']} — training data is not strictly in the past"
        )


def test_backtest_is_invariant_to_input_row_order(labeled):
    """A shuffled DataFrame must produce the same folds as a sorted one — the
    function must not silently trust caller ordering."""
    shuffled = labeled.sample(frac=1.0, random_state=7).reset_index(drop=True)

    y_true_sorted, y_pred_sorted, _, meta_sorted = walk_forward_backtest(labeled)
    y_true_shuffled, y_pred_shuffled, _, meta_shuffled = walk_forward_backtest(shuffled)

    assert meta_sorted == meta_shuffled
    np.testing.assert_array_equal(y_true_sorted, y_true_shuffled)
    np.testing.assert_allclose(y_pred_sorted, y_pred_shuffled)


def test_leakage_guard_assertion_actually_trips_on_a_duplicate_timestamp_boundary():
    """Proves the in-function assertion is load-bearing by actually calling
    walk_forward_backtest, not re-implementing its check. Engineers a fold boundary
    to land inside a run of duplicate timestamps, so some rows sharing one instant
    end up on both sides of the train/test split — exactly why upstream raw-data
    validation independently rejects duplicate timestamps."""
    rng = np.random.default_rng(0)
    n = 20
    timestamps = list(pd.date_range("2004-01-01", periods=n, freq="10min"))
    # Rows 6..9 all share row 6's timestamp, straddling where fold 1's train/test
    # boundary (index 8, from MIN_TRAIN_FRACTION=0.4 * n=20) falls.
    for i in range(7, 10):
        timestamps[i] = timestamps[6]
    df = pd.DataFrame({
        "timestamp": timestamps,
        "RUL": np.arange(n)[::-1],
        **{f: rng.normal(size=n) for f in FEATURE_NAMES},
    })

    with pytest.raises(AssertionError, match="leakage guard tripped"):
        walk_forward_backtest(df)


def test_rul_target_is_not_present_in_the_feature_matrix():
    assert "RUL" not in FEATURE_NAMES


def test_no_duplicate_timestamps_cross_a_fold_boundary(labeled):
    _, _, _, fold_meta = walk_forward_backtest(labeled)
    boundary_timestamps = {f["train_end"] for f in fold_meta} | {f["test_start"] for f in fold_meta}
    all_timestamps = pd.to_datetime(labeled["timestamp"])
    for ts in boundary_timestamps:
        assert (all_timestamps == pd.Timestamp(ts)).sum() <= 1, f"duplicate snapshot at fold boundary {ts}"
