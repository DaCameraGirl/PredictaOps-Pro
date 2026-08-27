"""Local artifact storage for registered-but-not-served ML models."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_ROOT = ROOT / "data" / "ml_models"


def model_root() -> Path:
    return Path(os.environ.get("PMS_MODEL_REGISTRY_ROOT", DEFAULT_MODEL_ROOT))


class ModelArtifactStore:
    def __init__(self, root: Path | None = None):
        self.root = root or model_root()

    def write_model(self, *, organization_id: str, experiment_run_id: str, model: Any) -> tuple[str, str]:
        directory = self.root / organization_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{experiment_run_id}.joblib"
        joblib.dump(model, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return path.as_posix(), digest

