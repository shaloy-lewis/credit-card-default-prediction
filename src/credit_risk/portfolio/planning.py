"""Standalone planning estimates for two independent, equally allocated binary arms.

Uses only Python's standard library. The normal approximation is equivalent to
R stats::power.prop.test(alternative="two.sided", strict=FALSE), with no
continuity correction. See docs/portfolio/planning-calculator.md for assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from fractions import Fraction
from statistics import NormalDist

METHOD = "equal_allocation_two_proportion_normal_strict_false"
MAX_PER_ARM = 2**53 - 1  # Counts must remain exactly representable during inversion.
NORMAL = NormalDist()


class PlanningError(ValueError):
    """Invalid or numerically unattainable planning request."""


@dataclass(frozen=True)
class PlanningEstimate:
    """All rates are fractions; participant counts are per customer, not account."""

    calculation: str
    baseline_rate: float
    treatment_rate: float
    absolute_reduction: float
    reduction_percentage_points: float
    relative_reduction: float
    alpha: float
    power: float
    attrition: float
    continuous_required_per_arm: float
    analysable_per_arm: int
    recruited_per_arm: int
    total_recruitment: int
    warnings: tuple[str, ...]
    label: str = "planning_estimate_not_observed_effect"
    method: str = METHOD
    allocation: str = "1:1 independent customers"


def _number(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PlanningError(f"{name} must be a finite number.")
    return float(value)


def _validate(baseline: float, alpha: float, power: float, attrition: float) -> None:
    for name, value in (("baseline_rate", baseline), ("alpha", alpha), ("power", power)):
        if not 0 < _number(name, value) < 1:
            raise PlanningError(f"{name} must be strictly between 0 and 1.")
    if power <= 0.5:
        raise PlanningError("power must be greater than 0.5 for this planning tool.")
    if not 0 <= _number("attrition", attrition) < 1:
        raise PlanningError("attrition must be at least 0 and below 1.")
    if not 0 < 1 - alpha / 2 < 1:
        raise PlanningError("alpha is too small for floating-point normal quantiles.")


def _required(baseline: float, reduction: float, alpha: float, power: float) -> float:
    """Continuous per-arm n from the normal-approximation power equation."""
    if reduction == 0:
        return math.inf
    treatment = baseline - reduction
    pooled = (baseline + treatment) / 2
    numerator = NORMAL.inv_cdf(1 - alpha / 2) * math.sqrt(2 * pooled * (1 - pooled))
    numerator += NORMAL.inv_cdf(power) * math.sqrt(
        baseline * (1 - baseline) + treatment * (1 - treatment)
    )
    try:
        return (numerator / reduction) ** 2
    except OverflowError:
        return math.inf


def _estimate(
    calculation: str,
    baseline: float,
    reduction: float,
    alpha: float,
    power: float,
    attrition: float,
    continuous: float,
    analysable: int,
    recruited: int,
) -> PlanningEstimate:
    treatment = baseline - reduction
    warnings = []
    if min(analysable * p for p in (baseline, 1 - baseline, treatment, 1 - treatment)) < 10:
        warnings.append("Expected events or non-events below 10: normal approximation is weak.")
    if attrition:
        warnings.append("Attrition inflation does not correct bias from missing outcomes.")
    return PlanningEstimate(
        calculation=calculation,
        baseline_rate=baseline,
        treatment_rate=treatment,
        absolute_reduction=reduction,
        reduction_percentage_points=100 * reduction,
        relative_reduction=reduction / baseline,
        alpha=alpha,
        power=power,
        attrition=attrition,
        continuous_required_per_arm=continuous,
        analysable_per_arm=analysable,
        recruited_per_arm=recruited,
        total_recruitment=2 * recruited,
        warnings=tuple(warnings),
    )


def sample_size(
    baseline_rate: float,
    absolute_reduction: float,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    attrition: float = 0.0,
) -> PlanningEstimate:
    """Round analysable n up, then inflate each arm for assumed attrition."""
    _validate(baseline_rate, alpha, power, attrition)
    reduction = _number("absolute_reduction", absolute_reduction)
    if not 0 < reduction <= baseline_rate:
        raise PlanningError("absolute_reduction must be positive and no larger than baseline_rate.")
    continuous = _required(baseline_rate, reduction, alpha, power)
    if not math.isfinite(continuous) or continuous > MAX_PER_ARM:
        raise PlanningError("Required sample size exceeds the supported numerical range.")
    analysable = max(2, math.ceil(continuous))
    # Preserve decimal attrition at integer boundaries (for example 0.9).
    inflated = math.ceil(Fraction(analysable) / (1 - Fraction(str(attrition))))
    if inflated > MAX_PER_ARM:
        raise PlanningError("Attrition-adjusted recruitment exceeds the supported numerical range.")
    return _estimate(
        "sample-size",
        baseline_rate,
        reduction,
        alpha,
        power,
        attrition,
        continuous,
        analysable,
        inflated,
    )


def minimum_detectable_effect(
    baseline_rate: float,
    n_per_arm: int,
    *,
    alpha: float = 0.05,
    power: float = 0.80,
    attrition: float = 0.0,
) -> PlanningEstimate:
    """Smallest detectable reduction, using floor(n_recruited * retention)."""
    _validate(baseline_rate, alpha, power, attrition)
    if (
        isinstance(n_per_arm, bool)
        or not isinstance(n_per_arm, int)
        or not 2 <= n_per_arm <= MAX_PER_ARM
    ):
        raise PlanningError("n_per_arm must be an integer from 2 through 2**53 - 1.")
    analysable = math.floor(n_per_arm * (1 - Fraction(str(attrition))))
    if analysable < 2:
        raise PlanningError("Attrition leaves fewer than two analysable customers per arm.")
    if _required(baseline_rate, baseline_rate, alpha, power) > analysable:
        raise PlanningError(
            "Target power is unattainable even if the treatment event rate is zero."
        )
    low, high = 0.0, baseline_rate
    for _ in range(100):
        mid = (low + high) / 2
        if _required(baseline_rate, mid, alpha, power) > analysable:
            low = mid
        else:
            high = mid
    return _estimate(
        "mde",
        baseline_rate,
        high,
        alpha,
        power,
        attrition,
        _required(baseline_rate, high, alpha, power),
        analysable,
        n_per_arm,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Planning estimates only; no model or data access."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("sample-size", "mde"):
        command = commands.add_parser(name)
        command.add_argument(
            "--baseline-rate", type=float, required=True, help="Fraction in (0, 1)."
        )
        command.add_argument("--alpha", type=float, default=0.05)
        command.add_argument("--power", type=float, default=0.80)
        command.add_argument(
            "--attrition", type=float, default=0.0, help="Expected missing fraction."
        )
        command.add_argument(
            "--json", action="store_true", help="Emit one deterministic JSON object."
        )
        if name == "sample-size":
            command.add_argument(
                "--absolute-reduction",
                type=float,
                required=True,
                help="Fraction, not percentage points.",
            )
        else:
            command.add_argument(
                "--n-per-arm",
                type=int,
                required=True,
                help="Recruited customers per arm, before attrition.",
            )
    args = parser.parse_args(argv)
    try:
        options = {"alpha": args.alpha, "power": args.power, "attrition": args.attrition}
        result = (
            sample_size(args.baseline_rate, args.absolute_reduction, **options)
            if args.command == "sample-size"
            else minimum_detectable_effect(args.baseline_rate, args.n_per_arm, **options)
        )
    except PlanningError as error:
        parser.error(str(error))
    if args.json:
        print(json.dumps(asdict(result), sort_keys=True, allow_nan=False))
    else:
        print("PLANNING ESTIMATE — not an observed or promised intervention effect")
        for key, value in asdict(result).items():
            print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
