"""Frozen validation-derived risk-band policy for release scoring."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

RiskBand = Literal["standard", "elevated", "high", "critical"]


class RiskPolicyError(ValueError):
    """Raised when a score or threshold set violates the release policy."""


def risk_band(probability: float, thresholds: Mapping[str, float]) -> RiskBand:
    """Assign one exclusive risk band using the frozen q80/q90/q95 cutoffs."""

    if not math.isfinite(probability) or probability < 0.0 or probability > 1.0:
        raise RiskPolicyError("Probability must be finite and within [0, 1].")
    if set(thresholds) != {"q80", "q90", "q95"}:
        raise RiskPolicyError("Risk thresholds must contain q80, q90, and q95.")
    q80, q90, q95 = (float(thresholds[name]) for name in ("q80", "q90", "q95"))
    if (
        not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (q80, q90, q95))
        or not q80 <= q90 <= q95
    ):
        raise RiskPolicyError("Risk thresholds must be finite, bounded, and ordered.")
    if probability >= q95:
        return "critical"
    if probability >= q90:
        return "high"
    if probability >= q80:
        return "elevated"
    return "standard"
