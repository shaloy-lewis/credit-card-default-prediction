"""Reference-fixed histograms; no fitting, hypothesis tests or automatic actions."""

from __future__ import annotations

from typing import Any

import numpy as np

from credit_risk.assurance.evidence import EvidenceError

MIN_ROWS = 200
WARNING = 0.10
INVESTIGATE = 0.20


def profile(values: Any, categorical: bool = False) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise EvidenceError("Reference needs non-empty finite observations.")
    if categorical:
        if not np.equal(array, np.rint(array)).all() or np.any((array < -2) | (array > 9)):
            raise EvidenceError("Repayment codes must be integers between -2 and 9.")
        cuts = [float(x) + 0.5 for x in range(-3, 10)]
    else:
        cuts = np.unique(np.quantile(array, np.linspace(0, 1, 11))).tolist()
        upper = float(np.nextafter(cuts[-1], np.inf))
        if np.isfinite(upper):
            cuts.append(upper)
        elif len(cuts) == 1:
            cuts.insert(0, float(np.nextafter(cuts[0], -np.inf)))
    counts = np.histogram(array, bins=[-np.inf, *cuts, np.inf])[0]
    return {
        "kind": "repayment" if categorical else "numeric",
        "cuts": cuts,
        "counts": counts.tolist(),
        "rows": len(array),
    }


def compare(reference: dict[str, Any], values: Any) -> dict[str, Any]:
    cuts = np.asarray(reference["cuts"], dtype=float)
    expected = np.asarray(reference["counts"], dtype=float)
    if (
        cuts.ndim != 1
        or not len(cuts)
        or not np.isfinite(cuts).all()
        or np.any(np.diff(cuts) <= 0)
        or expected.shape != (len(cuts) + 1,)
        or not np.isfinite(expected).all()
        or np.any(expected < 0)
        or expected.sum() != reference["rows"]
        or expected.sum() <= 0
    ):
        raise EvidenceError("Invalid reference histogram.")
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not np.isfinite(array).all():
        raise EvidenceError("Monitoring observations must be finite.")
    if not len(array):
        return {"rows": 0, "distance": None, "status": "insufficient_data"}
    observed = np.histogram(array, bins=[-np.inf, *cuts, np.inf])[0]
    distance = float(0.5 * np.abs(expected / expected.sum() - observed / len(array)).sum())
    distance = round(distance, 12)
    state = (
        "investigate" if distance >= INVESTIGATE else "warning" if distance >= WARNING else "clear"
    )
    return {
        "rows": len(array),
        "distance": distance,
        "status": state if len(array) >= MIN_ROWS else "insufficient_data",
    }
