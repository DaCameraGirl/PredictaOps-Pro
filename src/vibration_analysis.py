"""Time-domain waveform + FFT spectrum for a single bearing snapshot, with basic
data-quality checks. This is signal processing on the real captured window, not a
derived/simulated signal — the only transform applied before the FFT is a standard
Hann window (reduces spectral leakage from the window edges; doesn't add content).

We deliberately do NOT report a specific bearing-defect frequency (e.g. BPFO/BPFI).
That calculation needs shaft speed and bearing geometry (ball diameter, pitch
diameter, contact angle, number of rolling elements), none of which are provided
with this dataset, so any number we'd show would be invented, not derived.
"""
import numpy as np

from waveform_cache import WaveformCache


def _quality_warnings(signal: np.ndarray) -> list[str]:
    warnings = []
    std = float(signal.std())
    if std < 1e-6:
        warnings.append("flat signal (near-zero variance) — sensor may be inactive")
    peak = max(abs(signal.max()), abs(signal.min()))
    if peak > 0:
        clipped_fraction = float(np.mean(np.isclose(np.abs(signal), peak, rtol=1e-3)))
        if clipped_fraction > 0.05:
            warnings.append(f"possible clipping — {clipped_fraction * 100:.1f}% of samples near peak amplitude")
    return warnings


def analyze(cache: WaveformCache, bearing: str, timestamp_iso: str) -> dict:
    signal = cache.waveform(bearing, timestamp_iso)
    n = len(signal)
    dt = 1.0 / cache.sampling_rate_hz
    time_axis = (np.arange(n) * dt).tolist()

    windowed = signal * np.hanning(n)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(n, d=dt)
    magnitude = np.abs(spectrum) * 2 / n  # normalized amplitude, not power

    return {
        "bearing": bearing,
        "timestamp": timestamp_iso,
        "sampling_rate_hz": cache.sampling_rate_hz,
        "window_samples": n,
        "window_seconds": n * dt,
        "time_domain": {"time_s": time_axis, "amplitude": signal.tolist()},
        "fft": {"freq_hz": freqs.tolist(), "magnitude": magnitude.tolist()},
        "quality_warnings": _quality_warnings(signal),
        "note": (
            "FFT computed over a 2048-sample window at 20kHz (~102ms), not the full "
            "1-second capture. No bearing-defect frequency is reported: that requires "
            "shaft speed and bearing geometry, which this dataset does not provide."
        ),
    }
