"""RUL labeling and feature selection for the C-MAPSS dataset."""
import pandas as pd

from data_loader import SENSOR_COLS, SETTING_COLS
from profiling import profile_columns

RUL_CLIP = 125  # engines run near-flat early in life; cap so training targets aren't dominated by long healthy stretches


def signal_feature_cols(train_df: pd.DataFrame) -> list[str]:
    """Sensors/settings that actually vary, per the profiler. Constant columns carry no signal."""
    profile = profile_columns(train_df)
    candidates = SETTING_COLS + SENSOR_COLS
    return [c for c in candidates if not profile.loc[c, "constant"]]


def add_train_rul(df: pd.DataFrame, clip: int = RUL_CLIP) -> pd.DataFrame:
    """Attach a RUL label to every row of the training set (cycles until that unit's failure)."""
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - df["cycle"]).clip(upper=clip)
    return df.assign(RUL=rul)


def last_cycle_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent snapshot for each unit, used to score the test set against RUL_FD001.txt."""
    return df.loc[df.groupby("unit")["cycle"].idxmax()].reset_index(drop=True)
