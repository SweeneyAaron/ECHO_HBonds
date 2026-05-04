"""Precomputed ChemEM HBond normalization bounds."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from .reference_hbond_score import DATA_DIR, eval_poly, load_tables


BOUNDS_PATH = DATA_DIR / "hbond_score_bounds.json"
NORMALIZATION_MODE = "favorable_strength"
ANGLE_DOMAIN_MIN = 110.000001
ANGLE_DOMAIN_MAX = 180.0
DISTANCE_DOMAIN_MIN = 2.0
DISTANCE_DOMAIN_MAX = 5.999999


def load_score_bounds(path: str | Path = BOUNDS_PATH) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def get_pair_bound(
    donor_type: int,
    acceptor_type: int,
    bounds: Mapping[str, Any] | None = None,
) -> Mapping[str, float]:
    data = bounds or load_score_bounds()
    return data["pairs"][str(donor_type)][str(acceptor_type)]


def normalize_hbond_score(
    hbond_score: float,
    donor_type: int,
    acceptor_type: int,
    bounds: Mapping[str, Any] | None = None,
) -> float:
    pair_bound = get_pair_bound(donor_type, acceptor_type, bounds)
    max_favorable = float(pair_bound["max_favorable_magnitude"])
    if max_favorable <= 0.0:
        return 0.0
    return min(1.0, max(0.0, -float(hbond_score) / max_favorable))


def compute_all_score_bounds() -> dict[str, Any]:
    """Compute all donor/acceptor favorable-strength normalization bounds."""

    tables = load_tables()
    donor_keys = sorted(tables.poly_a.keys(), key=int)
    pairs: dict[str, dict[str, dict[str, float]]] = {}
    for donor_key in donor_keys:
        pairs[donor_key] = {}
        acceptor_keys = sorted(tables.poly_a[donor_key].keys(), key=int)
        for acceptor_key in acceptor_keys:
            coeff_a = tables.poly_a[donor_key][acceptor_key]
            coeff_b = tables.poly_b[donor_key][acceptor_key]
            coeff_c = tables.poly_c[donor_key][acceptor_key]
            pairs[donor_key][acceptor_key] = compute_pair_score_bound(
                coeff_a,
                coeff_b,
                coeff_c,
            )

    return {
        "normalization_mode": NORMALIZATION_MODE,
        "domain": {
            "angle_cutoff_degrees": 110.0,
            "angle_min_degrees": ANGLE_DOMAIN_MIN,
            "angle_max_degrees": ANGLE_DOMAIN_MAX,
            "distance_cutoff_angstrom": 6.0,
            "distance_min_angstrom": DISTANCE_DOMAIN_MIN,
            "distance_max_angstrom": DISTANCE_DOMAIN_MAX,
            "distance_clamp_angstrom": 2.0,
        },
        "pairs": pairs,
    }


def compute_pair_score_bound(
    coeff_a: Sequence[float],
    coeff_b: Sequence[float],
    coeff_c: Sequence[float],
) -> dict[str, float]:
    """Find the strongest favorable score for one polynomial donor/acceptor pair."""

    def score(angle: float, distance: float) -> float:
        return _hbond_score_from_coefficients(coeff_a, coeff_b, coeff_c, angle, distance)

    best_score = float("inf")
    best_angle = ANGLE_DOMAIN_MIN
    best_distance = DISTANCE_DOMAIN_MIN

    angle = ANGLE_DOMAIN_MIN
    while angle <= ANGLE_DOMAIN_MAX:
        distance = DISTANCE_DOMAIN_MIN
        while distance <= DISTANCE_DOMAIN_MAX:
            value = score(angle, distance)
            if value < best_score:
                best_score = value
                best_angle = angle
                best_distance = distance
            distance += 0.05
        angle += 1.0

    for _ in range(6):
        angle_lo = max(ANGLE_DOMAIN_MIN, best_angle - 2.0)
        angle_hi = min(ANGLE_DOMAIN_MAX, best_angle + 2.0)
        distance_lo = max(DISTANCE_DOMAIN_MIN, best_distance - 0.2)
        distance_hi = min(DISTANCE_DOMAIN_MAX, best_distance + 0.2)
        best_distance_score, best_distance = _ternary_min(
            lambda r: score(best_angle, r),
            distance_lo,
            distance_hi,
        )
        best_angle_score, best_angle = _ternary_min(
            lambda a: score(a, best_distance),
            angle_lo,
            angle_hi,
        )
        best_score = min(best_score, best_distance_score, best_angle_score)

    best_score = score(best_angle, best_distance)

    if best_score < 0.0:
        max_favorable = -best_score
        strongest = best_score
    else:
        max_favorable = 0.0
        strongest = best_score

    return {
        "max_favorable_magnitude": max_favorable,
        "strongest_favorable_energy": strongest,
        "angle_at_strongest_favorable": best_angle,
        "distance_at_strongest_favorable": best_distance,
    }


def _hbond_score_from_coefficients(
    coeff_a: Sequence[float],
    coeff_b: Sequence[float],
    coeff_c: Sequence[float],
    angle: float,
    distance: float,
) -> float:
    a_value = eval_poly(coeff_a, angle)
    b_value = eval_poly(coeff_b, angle)
    c_value = eval_poly(coeff_c, angle)
    r_clamped = max(distance, 2.0)
    r2 = r_clamped * r_clamped
    r6 = r2 * r2 * r2
    return a_value * math.exp(-b_value * r_clamped) - c_value / r6


def _ternary_min(function, low: float, high: float, iterations: int = 80) -> tuple[float, float]:
    for _ in range(iterations):
        first = low + (high - low) / 3.0
        second = high - (high - low) / 3.0
        if function(first) < function(second):
            high = second
        else:
            low = first
    point = (low + high) / 2.0
    return function(point), point
