"""Deterministic unit normalization for canonical machine data."""

from __future__ import annotations


class UnitNormalizationError(ValueError):
    pass


def _key(unit: str) -> str:
    return unit.strip().lower().replace(" ", "")


def normalize_unit_value(value: float, from_unit: str, to_unit: str) -> tuple[float, str]:
    source = _key(from_unit)
    target = _key(to_unit)
    if source == target:
        return value, to_unit

    conversions = {
        ("f", "c"): lambda x: (x - 32.0) * 5.0 / 9.0,
        ("c", "f"): lambda x: (x * 9.0 / 5.0) + 32.0,
        ("k", "c"): lambda x: x - 273.15,
        ("c", "k"): lambda x: x + 273.15,
        ("m/s^2", "g"): lambda x: x / 9.80665,
        ("m/s2", "g"): lambda x: x / 9.80665,
        ("g", "m/s^2"): lambda x: x * 9.80665,
        ("g", "m/s2"): lambda x: x * 9.80665,
        ("hz", "rpm"): lambda x: x * 60.0,
        ("rpm", "hz"): lambda x: x / 60.0,
    }
    converter = conversions.get((source, target))
    if converter is None:
        raise UnitNormalizationError(f"unsupported unit conversion from {from_unit!r} to {to_unit!r}")
    return converter(value), to_unit

