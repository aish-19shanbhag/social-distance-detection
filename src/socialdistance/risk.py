"""Pairwise ground distances and proximity risk classification.

Thresholds (same as the 2021 project):

==========  ======================  ======
Risk        Pair distance           Colour
==========  ======================  ======
High        d <= 150 cm             red
Low         150 cm < d <= 180 cm    yellow
None        d > 180 cm              green
==========  ======================  ======

Each person takes the **worst** risk over all pairs they are part of, so someone who is
150 cm from one person and 4 m from everyone else is high risk. A person alone in the
frame is no risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class RiskLevel(IntEnum):
    HIGH = 0
    LOW = 1
    NONE = 2

    @property
    def label(self) -> str:
        return {RiskLevel.HIGH: "high risk", RiskLevel.LOW: "low risk", RiskLevel.NONE: "no risk"}[self]


@dataclass(frozen=True)
class RiskThresholds:
    high_cm: float = 150.0
    low_cm: float = 180.0

    def __post_init__(self) -> None:
        if not 0 < self.high_cm <= self.low_cm:
            raise ValueError("Thresholds must satisfy 0 < high_cm <= low_cm")

    def classify(self, distance_cm: float) -> RiskLevel:
        if distance_cm <= self.high_cm:
            return RiskLevel.HIGH
        if distance_cm <= self.low_cm:
            return RiskLevel.LOW
        return RiskLevel.NONE


@dataclass
class RiskAssessment:
    distances: np.ndarray  # (N, N) pairwise ground distances in cm
    pairs: list[tuple[int, int, float, RiskLevel]]  # i < j, distance, level
    person_levels: list[RiskLevel]
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def min_distance_cm(self) -> float | None:
        if not self.pairs:
            return None
        return float(min(d for _, _, d, _ in self.pairs))

    @property
    def violations(self) -> list[tuple[int, int, float, RiskLevel]]:
        """Pairs closer than the low-risk threshold."""
        return [p for p in self.pairs if p[3] != RiskLevel.NONE]


def pairwise_distances(points_cm: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_cm, dtype=np.float64).reshape(-1, 2)
    diff = pts[:, None, :] - pts[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def count_levels(levels: list[RiskLevel]) -> dict[str, int]:
    return {
        "high": sum(1 for lv in levels if lv == RiskLevel.HIGH),
        "low": sum(1 for lv in levels if lv == RiskLevel.LOW),
        "safe": sum(1 for lv in levels if lv == RiskLevel.NONE),
        "total": len(levels),
    }


def assess_risk(points_cm: np.ndarray, thresholds: RiskThresholds | None = None) -> RiskAssessment:
    """Classify every pair and every person from ground positions in centimetres."""
    thresholds = thresholds or RiskThresholds()
    dist = pairwise_distances(points_cm)
    n = len(dist)
    levels = [RiskLevel.NONE] * n
    pairs: list[tuple[int, int, float, RiskLevel]] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist[i, j])
            level = thresholds.classify(d)
            pairs.append((i, j, d, level))
            levels[i] = min(levels[i], level)
            levels[j] = min(levels[j], level)
    return RiskAssessment(distances=dist, pairs=pairs, person_levels=levels, counts=count_levels(levels))
