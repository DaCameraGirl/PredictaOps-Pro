"""Waveform loading with deterministic integrity checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from platform_core.models import WaveformRecord


class WaveformIntegrityError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedWaveform:
    samples: np.ndarray
    content_sha256: str
    checksum_verified: bool


def load_waveform(record: WaveformRecord) -> LoadedWaveform:
    path = _local_path(record.storage_uri)
    if path is None:
        raise WaveformIntegrityError("waveform content is external and not locally available for analytics")
    if not path.exists():
        raise WaveformIntegrityError("waveform content is missing from storage")

    raw = path.read_bytes()
    content_sha256 = hashlib.sha256(raw).hexdigest()
    checksum_verified = False
    if record.sha256:
        checksum_verified = content_sha256 == record.sha256
        if not checksum_verified:
            raise WaveformIntegrityError("waveform content checksum does not match provenance")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WaveformIntegrityError("waveform content is not valid JSON") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
        raise WaveformIntegrityError("waveform content does not contain a samples array")

    try:
        samples = np.asarray(payload["samples"], dtype=float)
    except (TypeError, ValueError) as exc:
        raise WaveformIntegrityError("waveform samples are not numeric") from exc
    if samples.size != record.sample_count:
        raise WaveformIntegrityError("waveform sample_count does not match stored content")
    if not np.isfinite(samples).all():
        raise WaveformIntegrityError("waveform samples contain non-finite values")
    return LoadedWaveform(samples=samples, content_sha256=content_sha256, checksum_verified=checksum_verified)


def _local_path(storage_uri: str) -> Path | None:
    parsed = urlparse(storage_uri)
    if parsed.scheme in ("", None):
        return Path(storage_uri)
    if parsed.scheme == "file":
        return Path(parsed.path)
    if len(parsed.scheme) == 1 and storage_uri[1:3] in (":/", ":\\"):
        return Path(storage_uri)
    return None

