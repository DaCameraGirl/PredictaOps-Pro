"""Local landing storage for raw waveform payloads."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WAVEFORM_ROOT = ROOT / "data" / "waveforms"


def waveform_root() -> Path:
    return Path(os.environ.get("PMS_WAVEFORM_ROOT", DEFAULT_WAVEFORM_ROOT))


class LocalWaveformStore:
    def __init__(self, root: Path | None = None):
        self.root = root or waveform_root()

    def put_samples(
        self,
        *,
        organization_id: str,
        batch_id: str,
        record_key: str,
        samples: list[float],
        metadata: dict[str, Any],
    ) -> tuple[str, str, int]:
        payload = {"samples": samples, "metadata": metadata}
        raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        directory = self.root / organization_id / batch_id
        directory.mkdir(parents=True, exist_ok=True)
        object_key = hashlib.sha256(record_key.encode("utf-8")).hexdigest()
        path = directory / f"{object_key}.json"
        path.write_bytes(raw)
        return path.as_posix(), digest, len(samples)

    def describe_external(
        self,
        *,
        storage_uri: str,
        sha256: str | None,
        sample_count: int,
    ) -> tuple[str, str | None, int]:
        return storage_uri, sha256, sample_count
