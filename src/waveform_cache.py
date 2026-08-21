"""Compact per-snapshot raw waveform sample, for the time-domain/FFT views.

Serving the full raw dataset (~525MB, 984 files x 20480 samples x 4 channels) to a
deployed app isn't practical, and it's excluded from git for the same reason. Instead
we keep the first WINDOW_SAMPLES of each snapshot at the original 20kHz sample rate
(a short, real, contiguous slice, not a decimated/resampled approximation) for every
snapshot and every bearing, compressed into one small array file that ships with the
app. That's enough for an honest time-domain waveform and FFT spectrum at every point
on the timeline, at the cost of a shorter time window than the full 1-second capture.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from bearing_data import BEARING_COLS, EXPECTED_SAMPLING_RATE_HZ, PROCESSED_DIR, RAW_DIR, _parse_timestamp, _raw_files

WINDOW_SAMPLES = 2048  # ~102ms at 20kHz; FFT bin width ~9.77Hz
CACHE_PATH = PROCESSED_DIR / "waveform_cache.npz"


def build_waveform_cache(raw_dir: Path = RAW_DIR) -> None:
    files = _raw_files(raw_dir)
    timestamps = [_parse_timestamp(p.name).isoformat() for p in files]
    per_bearing = {b: np.empty((len(files), WINDOW_SAMPLES), dtype=np.float32) for b in BEARING_COLS}

    for row_i, path in enumerate(files):
        snap = pd.read_csv(path, sep=r"\s+", header=None, nrows=WINDOW_SAMPLES).to_numpy(dtype=np.float32)
        for col_i, bearing in enumerate(BEARING_COLS):
            per_bearing[bearing][row_i] = snap[:, col_i]

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        CACHE_PATH,
        timestamps=np.array(timestamps),
        sampling_rate_hz=EXPECTED_SAMPLING_RATE_HZ,
        window_samples=WINDOW_SAMPLES,
        **per_bearing,
    )
    print(f"wrote waveform cache to {CACHE_PATH} ({CACHE_PATH.stat().st_size / 1e6:.1f} MB)")


class WaveformCache:
    def __init__(self, path: Path = CACHE_PATH):
        data = np.load(path)
        self.timestamps = list(data["timestamps"])
        self.sampling_rate_hz = int(data["sampling_rate_hz"])
        self.window_samples = int(data["window_samples"])
        self._by_bearing = {b: data[b] for b in BEARING_COLS}
        self._index = {ts: i for i, ts in enumerate(self.timestamps)}

    def waveform(self, bearing: str, timestamp_iso: str) -> np.ndarray:
        return self._by_bearing[bearing][self._index[timestamp_iso]]


if __name__ == "__main__":
    build_waveform_cache()
