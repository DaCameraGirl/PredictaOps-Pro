import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from bearing_data import DEFAULT_RUN, FEATURES_CACHE, load_feature_table  # noqa: E402


@pytest.fixture(scope="session")
def feature_table():
    if not FEATURES_CACHE.exists():
        pytest.skip(f"processed features not found at {FEATURES_CACHE}; run src/train_bearing.py first")
    return load_feature_table(DEFAULT_RUN)


@pytest.fixture(scope="session")
def model_dir():
    path = ROOT / "models"
    if not (path / "bearing_rul_model.joblib").exists():
        pytest.skip("trained model not found; run src/train_bearing.py first")
    return path
