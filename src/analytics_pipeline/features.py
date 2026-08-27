"""Deterministic scalar and waveform feature extraction."""

from __future__ import annotations

import math

import numpy as np

from analytics_pipeline.contracts import FeatureValue


class FeatureExtractionError(ValueError):
    pass


def scalar_features(metric: str, value: float, unit: str | None) -> list[FeatureValue]:
    if not math.isfinite(value):
        raise FeatureExtractionError("scalar reading value must be finite")
    return [FeatureValue(name=f"scalar.{metric}", value=float(value), unit=unit)]


def waveform_features(samples: np.ndarray, *, unit: str | None, sampling_rate_hz: float) -> list[FeatureValue]:
    if samples.size == 0:
        raise FeatureExtractionError("waveform samples are empty")
    if sampling_rate_hz <= 0:
        raise FeatureExtractionError("waveform sampling_rate_hz must be positive")
    if not np.isfinite(samples).all():
        raise FeatureExtractionError("waveform samples must be finite")

    mean = float(np.mean(samples))
    std = float(np.std(samples))
    rms = float(np.sqrt(np.mean(np.square(samples))))
    max_value = float(np.max(samples))
    min_value = float(np.min(samples))
    peak_to_peak = float(max_value - min_value)
    peak_abs = float(np.max(np.abs(samples)))
    crest_factor = float(peak_abs / rms) if rms else 0.0
    kurtosis = _population_kurtosis(samples, mean, std)
    freqs, magnitudes = _fft_magnitudes(samples, sampling_rate_hz)
    dominant_frequency = _dominant_frequency(freqs, magnitudes)
    spectral_centroid = _spectral_centroid(freqs, magnitudes)

    return [
        FeatureValue(name="waveform.mean", value=mean, unit=unit),
        FeatureValue(name="waveform.std", value=std, unit=unit),
        FeatureValue(name="waveform.rms", value=rms, unit=unit),
        FeatureValue(name="waveform.min", value=min_value, unit=unit),
        FeatureValue(name="waveform.max", value=max_value, unit=unit),
        FeatureValue(name="waveform.peak_to_peak", value=peak_to_peak, unit=unit),
        FeatureValue(name="waveform.kurtosis", value=kurtosis, unit=None),
        FeatureValue(name="waveform.crest_factor", value=crest_factor, unit=None),
        FeatureValue(name="fft.dominant_frequency_hz", value=dominant_frequency, unit="Hz"),
        FeatureValue(name="fft.spectral_centroid_hz", value=spectral_centroid, unit="Hz"),
    ]


def _population_kurtosis(samples: np.ndarray, mean: float, std: float) -> float:
    if std == 0.0:
        return 0.0
    normalized = (samples - mean) / std
    return float(np.mean(np.power(normalized, 4)))


def _fft_magnitudes(samples: np.ndarray, sampling_rate_hz: float) -> tuple[np.ndarray, np.ndarray]:
    freqs = np.fft.rfftfreq(samples.size, d=1.0 / sampling_rate_hz)
    magnitudes = np.abs(np.fft.rfft(samples))
    return freqs, magnitudes


def _dominant_frequency(freqs: np.ndarray, magnitudes: np.ndarray) -> float:
    if magnitudes.size <= 1:
        return 0.0
    idx = int(np.argmax(magnitudes[1:]) + 1)
    return float(freqs[idx])


def _spectral_centroid(freqs: np.ndarray, magnitudes: np.ndarray) -> float:
    total = float(np.sum(magnitudes))
    if total == 0.0:
        return 0.0
    return float(np.sum(freqs * magnitudes) / total)

