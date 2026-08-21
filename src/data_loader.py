"""Load the NASA C-MAPSS FD001 turbofan degradation dataset."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

INDEX_COLS = ["unit", "cycle"]
SETTING_COLS = ["op_setting_1", "op_setting_2", "op_setting_3"]
SENSOR_COLS = [f"sensor_{i}" for i in range(1, 22)]
COLUMNS = INDEX_COLS + SETTING_COLS + SENSOR_COLS


def _read_space_separated(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None)
    df = df.iloc[:, : len(COLUMNS)]
    df.columns = COLUMNS
    return df


def load_train(dataset: str = "FD001") -> pd.DataFrame:
    return _read_space_separated(DATA_DIR / f"train_{dataset}.txt")


def load_test(dataset: str = "FD001") -> pd.DataFrame:
    return _read_space_separated(DATA_DIR / f"test_{dataset}.txt")


def load_test_rul(dataset: str = "FD001") -> pd.Series:
    """True remaining useful life for each unit's last test cycle."""
    path = DATA_DIR / f"RUL_{dataset}.txt"
    rul = pd.read_csv(path, sep=r"\s+", header=None)[0]
    rul.index = pd.RangeIndex(1, len(rul) + 1)
    rul.index.name = "unit"
    rul.name = "RUL"
    return rul
