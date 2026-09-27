"""Frozen stress grid; no training or hypothetical-outcome metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from credit_risk.assurance.evidence import (
    ROOT,
    EvidenceError,
    clean_commit,
    encode,
    hash_file,
    publish,
    read_json,
    safe_path,
    source_map,
    verify,
)
from credit_risk.assurance.validation import check_baseline, metrics, validation_cohort
from credit_risk.inference.contracts import OperationalFeatures
from credit_risk.inference.engine import InferenceEngine, InferenceResult
from credit_risk.modeling.contracts import REPAYMENT_STATUS_COLUMNS

CONFIG = "configs/robustness/release_b_v1.json"
KIND = "release_b_robustness_v1"
DEFAULT_OUTPUT = "reports/robustness/release_b_v1"
GROUPS = {
    "credit_capacity": ("credit_limit_ntd",),
    "billing_balance": tuple(f"bill_amount_ntd_lag_{i}" for i in range(6)),
    "payment_behaviour": tuple(f"payment_amount_ntd_lag_{i}" for i in range(6)),
}


def scenarios(features: pd.DataFrame) -> dict[str, pd.DataFrame]:
    result: dict[str, pd.DataFrame] = {}
    for group, columns in GROUPS.items():
        for factor in (0.9, 1.1, 0.75, 1.25):
            frame = features.copy()
            frame.loc[:, list(columns)] = np.rint(frame.loc[:, list(columns)] * factor).astype(
                "int64"
            )
            if group == "credit_capacity":
                frame["credit_limit_ntd"] = frame["credit_limit_ntd"].clip(lower=1)
            result[f"{group}_{factor:g}"] = frame
    for increment in (1, 2):
        frame = features.copy()
        values = frame.loc[:, list(REPAYMENT_STATUS_COLUMNS)]
        frame.loc[:, list(REPAYMENT_STATUS_COLUMNS)] = values.where(
            values < 0, (values + increment).clip(upper=9)
        )
        result[f"repayment_plus_{increment}"] = frame
    return result


def selected_ids(features: pd.DataFrame, probabilities: np.ndarray) -> set[Any]:
    order = np.lexsort((features.index.to_numpy(), -probabilities))
    return set(features.index[order[: int(len(features) * 0.1)]].tolist())


def sensitivity(
    features: pd.DataFrame, baseline: InferenceResult, changed: InferenceResult
) -> dict[str, float]:
    before = selected_ids(features, baseline.probabilities)
    after = selected_ids(features, changed.probabilities)
    difference = np.abs(changed.probabilities - baseline.probabilities)
    return {
        "mean_absolute_probability_change": float(np.mean(difference)),
        "max_absolute_probability_change": float(np.max(difference)),
        "risk_band_change_fraction": float(
            np.mean(np.array(baseline.risk_bands) != np.array(changed.risk_bands))
        ),
        "queue_turnover": len(before - after) / len(before) if before else 0.0,
        "reason_change_fraction": sum(
            tuple((r.category, r.direction) for r in left)
            != tuple((r.category, r.direction) for r in right)
            for left, right in zip(baseline.reasons, changed.reasons, strict=True)
        )
        / len(features),
        "max_additivity_error": changed.max_additivity_error,
        "max_probability_error": changed.max_probability_error,
    }


def material(values: dict[str, float]) -> bool:
    return (
        values["mean_absolute_probability_change"] >= 0.05
        or values["queue_turnover"] >= 0.2
        or values["risk_band_change_fraction"] >= 0.2
        or values["reason_change_fraction"] >= 0.2
    )


def build(
    data_root: str = "data",
    output: str = DEFAULT_OUTPUT,
    runtime: str = "experiment/robustness/release_b_v1",
) -> str:
    commit = clean_commit()
    destination = safe_path(output, "reports/robustness")
    runtime_path = safe_path(runtime, "experiment/robustness")
    if destination.exists() or runtime_path.exists():
        raise EvidenceError("Refusing to overwrite robustness evidence or runtime rows.")
    config = read_json(ROOT / CONFIG)
    if config != {
        "protocol_id": KIND,
        "monetary_multipliers": [0.9, 1.1, 0.75, 1.25],
        "repayment_increments": [1, 2],
        "subset_limit_quantile": 0.25,
        "subset_repayment_minimum": 2,
        "validation_rows": 4800,
        "material_mean_probability_change": 0.05,
        "material_queue_turnover": 0.2,
        "material_band_or_reason_change": 0.2,
        "disposition": "human_review_required",
        "perturbed_outcome_metrics": "prohibited",
    }:
        raise EvidenceError("Robustness protocol differs from the frozen grid.")
    sources = source_map(
        [
            CONFIG,
            "models/selected_v1/manifest.json",
            "configs/governance/phase5_v1.json",
            "configs/inference/phase6_v1.json",
            "configs/data/split_v1.lock.json",
        ]
    )
    features, target = validation_cohort(data_root)
    engine = InferenceEngine()
    baseline = engine.score(features)
    baseline_metrics = check_baseline(target.to_numpy(), baseline.probabilities)
    runtime_path.mkdir(parents=True)
    pd.DataFrame({"account_id": features.index, "probability": baseline.probabilities}).to_csv(
        runtime_path / "baseline.csv", index=False
    )
    results = []
    for name, frame in scenarios(features).items():
        for row in frame.to_dict("records"):
            OperationalFeatures.model_validate(row)
        scored = engine.score(frame)
        values = sensitivity(features, baseline, scored)
        pd.DataFrame({"account_id": frame.index, "probability": scored.probabilities}).to_csv(
            runtime_path / f"{name}.csv", index=False
        )
        results.append(
            {
                "scenario": name,
                "rows": len(frame),
                "sensitivity": values,
                "material": material(values),
                "disposition": "pending_owner_review",
                "outcome_metrics": "not_applicable_counterfactual_features",
            }
        )
    masks = {
        "low_credit_limit": features.credit_limit_ntd.le(features.credit_limit_ntd.quantile(0.25)),
        "repayment_at_least_two": features.loc[:, list(REPAYMENT_STATUS_COLUMNS)].ge(2).any(axis=1),
    }
    subsets = []
    for name, mask in masks.items():
        subsets.append(
            {
                "scenario": name,
                "metrics": metrics(target.loc[mask].to_numpy(), baseline.probabilities[mask]),
                "disposition": "pending_owner_review",
                "interpretation": "historical_subset_only",
            }
        )
    summary = {
        "status": "pending_owner_review",
        "validation_rows": 4800,
        "model_sha256": engine.config.bundle.model_sha256,
        "baseline": baseline_metrics,
        "scenarios": results,
        "subsets": subsets,
        "runtime_sha256": {p.name: hash_file(p) for p in sorted(runtime_path.iterdir())},
        "test_accounts_selected": False,
        "perturbed_outcome_metrics": False,
    }
    return publish(
        destination,
        kind=KIND,
        summary=summary,
        sources=sources,
        commit=commit,
        extra={"protocol.json": encode(config)},
    )


def verify_evidence(root: str, expected: str) -> dict[str, Any]:
    summary = verify(root, expected, KIND)
    expected_names = {
        f"{group}_{factor:g}" for group in GROUPS for factor in (0.9, 1.1, 0.75, 1.25)
    }
    expected_names |= {"repayment_plus_1", "repayment_plus_2"}
    actual = [item["scenario"] for item in summary["scenarios"]]
    if len(actual) != 14 or set(actual) != expected_names or summary["validation_rows"] != 4800:
        raise EvidenceError("Incomplete robustness scenarios.")
    if (
        summary["test_accounts_selected"] is not False
        or summary["perturbed_outcome_metrics"] is not False
    ):
        raise EvidenceError("Robustness evidence violated the outcome boundary.")
    return summary
