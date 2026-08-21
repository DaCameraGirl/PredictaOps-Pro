import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

import pandas as pd  # noqa: E402

from bearing_data import FEATURES_CACHE  # noqa: E402


@pytest.fixture(scope="session")
def feature_table():
    if not FEATURES_CACHE.exists():
        pytest.skip(f"processed features not found at {FEATURES_CACHE}; run src/train_bearing.py first")
    return pd.read_csv(FEATURES_CACHE, parse_dates=["timestamp"])


@pytest.fixture(scope="session")
def model_dir():
    path = ROOT / "models"
    if not (path / "bearing_rul_model.joblib").exists():
        pytest.skip("trained model not found; run src/train_bearing.py first")
    return path
