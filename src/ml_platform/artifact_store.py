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

    def _trusted_artifact_path(self, *, organization_id: str, artifact_uri: str) -> Path:
        try:
            path = Path(artifact_uri).resolve(strict=True)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"model artifact does not exist: {artifact_uri}") from exc
        trusted_root = (self.root / organization_id).resolve(strict=False)
        try:
            path.relative_to(trusted_root)
        except ValueError as exc:
            raise ValueError("model artifact path is outside the trusted registry root") from exc
        return path

    def write_model(self, *, organization_id: str, experiment_run_id: str, model: Any) -> tuple[str, str]:
        directory = self.root / organization_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{experiment_run_id}.joblib"
        joblib.dump(model, path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return path.as_posix(), digest

    def verify_artifact(self, *, organization_id: str, artifact_uri: str, expected_sha256: str) -> str:
        path = self._trusted_artifact_path(organization_id=organization_id, artifact_uri=artifact_uri)
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("model artifact SHA-256 does not match registry metadata")
        return actual_sha256

    def load_verified_model(self, *, organization_id: str, artifact_uri: str, expected_sha256: str) -> Any:
        self.verify_artifact(
            organization_id=organization_id,
            artifact_uri=artifact_uri,
            expected_sha256=expected_sha256,
        )
        path = self._trusted_artifact_path(organization_id=organization_id, artifact_uri=artifact_uri)
        return joblib.load(path)

