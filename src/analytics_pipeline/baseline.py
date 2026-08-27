"""Baseline, anomaly, and trend scoring for production Slice 8."""

from __future__ import annotations

import numpy as np

from platform_core.models import AnalyticsFeatureRecord

MIN_BASELINE_SAMPLES = 3


def score_feature(
    current: AnalyticsFeatureRecord,
    history: list[AnalyticsFeatureRecord],
    *,
    min_baseline_samples: int = MIN_BASELINE_SAMPLES,
) -> tuple[str, float | None, float | None, float, dict]:
    prior = [
        row
        for row in history
        if row.observed_at < current.observed_at and row.feature_name == current.feature_name
    ]
    if len(prior) < min_baseline_samples:
        return (
            "insufficient_evidence",
            None,
            _trend_slope([*prior, current]),
            0.0,
            {
                "reason": "baseline_window_too_small",
                "feature_name": current.feature_name,
                "baseline_samples": len(prior),
                "minimum_baseline_samples": min_baseline_samples,
            },
        )

    values = np.asarray([row.value for row in prior], dtype=float)
    mean = float(np.mean(values))
    std = float(np.std(values))
    effective_std = max(std, abs(mean) * 0.05, 1e-9)
    z_score = float(abs(current.value - mean) / effective_std)
    health_state = _state_from_score(z_score)
    return (
        health_state,
        z_score,
        _trend_slope([*prior[-5:], current]),
        _confidence(len(prior), z_score),
        {
            "feature_name": current.feature_name,
            "current_value": current.value,
            "baseline_mean": mean,
            "baseline_std": std,
            "effective_std": effective_std,
            "baseline_samples": len(prior),
            "anomaly_score": z_score,
            "thresholds": {"watch": 2.0, "warning": 4.0, "critical": 6.0},
        },
    )


def choose_health_state(
    scores: list[tuple[str, float | None, float | None, float, dict]],
) -> tuple[str, float | None, float | None, float, dict]:
    if not scores:
        return "unknown", None, None, 0.0, {"reason": "no_analytics_features"}
    scored = [score for score in scores if score[1] is not None]
    if not scored:
        state, anomaly_score, trend_slope, confidence, evidence = scores[0]
        return state, anomaly_score, trend_slope, confidence, evidence
    state, anomaly_score, trend_slope, confidence, evidence = max(scored, key=lambda item: item[1] or 0.0)
    return state, anomaly_score, trend_slope, confidence, evidence


def _state_from_score(z_score: float) -> str:
    if z_score >= 6.0:
        return "critical"
    if z_score >= 4.0:
        return "warning"
    if z_score >= 2.0:
        return "watch"
    return "healthy"


def _confidence(baseline_samples: int, z_score: float) -> float:
    sample_factor = min(1.0, baseline_samples / 10.0)
    signal_factor = min(1.0, max(0.2, z_score / 6.0))
    return float(round(sample_factor * signal_factor, 4))


def _trend_slope(rows: list[AnalyticsFeatureRecord]) -> float | None:
    if len(rows) < 3:
        return None
    values = np.asarray([row.value for row in rows], dtype=float)
    x = np.arange(values.size, dtype=float)
    return float(np.polyfit(x, values, deg=1)[0])
