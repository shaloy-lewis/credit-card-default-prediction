"""Tests for the strict Phase 3 candidate configuration contract."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from credit_risk.modeling.candidate_contracts import (
    CandidateExperimentConfig,
    load_candidate_config,
    parse_candidate_config,
)
from credit_risk.modeling.contracts import ModelingContractError

CONFIG_PATH = Path("configs/modeling/candidate_v1.json")


def _payload() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_checked_in_candidate_contract_resolves_all_reviewed_evidence() -> None:
    config = load_candidate_config()

    assert config.protocol_id == "candidate_v1"
    assert len(config.candidate.search.sampled_configurations) == 12
    assert config.candidate.search.maximum_fold_fits == 210
    assert [view.eligible_for_advancement for view in config.feature_views] == [
        False,
        False,
        True,
    ]


def test_candidate_parser_is_frozen_and_rejects_extra_fields() -> None:
    payload = _payload()
    payload["unreviewed"] = True

    with pytest.raises(ModelingContractError, match="Extra inputs are not permitted"):
        parse_candidate_config(json.dumps(payload))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (("data_contract", "holdout_access", "allowed"), "holdout_access"),
        (("feature_views", 0, "eligible_for_advancement", True), "feature views"),
        (("candidate", "additional_challenger", "xgboost"), "additional_challenger"),
        (("candidate", "fixed_parameters", "thread_count", 2), "thread_count"),
        (("candidate", "search", "maximum_fold_fits", 211), "maximum_fold_fits"),
        (
            ("candidate", "search", "parameter_space", "learning_rate", [0.02, 0.05, 0.1]),
            "parameter space",
        ),
        (
            (
                "candidate",
                "search",
                "ablation_policy",
                "evaluation_feature_views",
                ["monetary_only", "repayment_status_only", "operational_full"],
            ),
            "ablation views",
        ),
        (("evaluation", "primary_capacity_metric", "capacity", 0.2), "10%"),
        (
            ("evaluation", "reported_discrimination_metrics", ["roc_auc"]),
            "discrimination metrics",
        ),
        (("evaluation", "reported_probability_metrics", ["brier_score"]), "probability metrics"),
        (("evaluation", "reported_capacities", [0.1]), "capacities"),
        (("evaluation", "repeat_summary", ["mean"]), "repeat summaries"),
        (
            (
                "advancement_gate",
                "relative_thresholds",
                "minimum_average_precision_improvement",
                0.02,
            ),
            "relative candidate thresholds",
        ),
        (("advancement_gate", "equivalence_band_average_precision", 0.01), "equivalence"),
        (("advancement_gate", "tie_break_order", ["lower_depth"]), "tie-break"),
        (
            ("advancement_gate", "final_variant_selection", "best_feature_view"),
            "final_variant_selection",
        ),
        (("governance", "deferred_to_phase_4", ["calibration_selection"]), "deferrals"),
    ),
)
def test_candidate_contract_rejects_protocol_drift(mutation, message: str) -> None:
    payload = copy.deepcopy(_payload())
    *parents, key, value = mutation
    target = payload
    for parent in parents:
        target = target[parent]  # type: ignore[index,assignment]
    target[key] = value  # type: ignore[index]

    with pytest.raises(ModelingContractError, match=message):
        parse_candidate_config(json.dumps(payload))


def test_candidate_contract_rejects_changed_materialized_sample() -> None:
    payload = _payload()
    sampled = payload["candidate"]["search"]["sampled_configurations"]  # type: ignore[index]
    sampled[0]["parameters"]["depth"] = 4

    with pytest.raises(ModelingContractError, match="sampler output"):
        parse_candidate_config(json.dumps(payload))


@pytest.mark.parametrize(
    ("columns", "count", "message"),
    (
        (["repayment_status_lag_0"], 6, "predictor count"),
        (["repayment_status_lag_0"] * 6, 6, "unique"),
        (["account_id", *[f"repayment_status_lag_{index}" for index in range(5)]], 6, "forbidden"),
    ),
)
def test_candidate_feature_views_reject_invalid_column_boundaries(
    columns: list[str],
    count: int,
    message: str,
) -> None:
    payload = _payload()
    view = payload["feature_views"][0]  # type: ignore[index]
    view["predictor_columns"] = columns
    view["predictor_count"] = count

    with pytest.raises(ModelingContractError, match=message):
        parse_candidate_config(json.dumps(payload))


def test_candidate_contract_rejects_feature_handling_and_absolute_gate_drift() -> None:
    payload = _payload()
    payload["candidate"]["feature_handling"]["numeric_columns"] = [  # type: ignore[index]
        "credit_limit_ntd"
    ]
    with pytest.raises(ModelingContractError, match="feature handling"):
        parse_candidate_config(json.dumps(payload))

    payload = _payload()
    payload["advancement_gate"]["derived_absolute_thresholds"][  # type: ignore[index]
        "minimum_average_precision_mean"
    ] = 0.9
    with pytest.raises(ModelingContractError, match="absolute gate thresholds"):
        parse_candidate_config(json.dumps(payload))


def test_candidate_contract_rejects_duplicate_or_reordered_sample_ids() -> None:
    payload = _payload()
    sampled = payload["candidate"]["search"]["sampled_configurations"]  # type: ignore[index]
    sampled[1]["configuration_id"] = "cb_cfg_001"

    with pytest.raises(ModelingContractError, match="cb_cfg_001..012"):
        parse_candidate_config(json.dumps(payload))


def test_candidate_contract_rejects_unsafe_reference_paths() -> None:
    payload = _payload()
    payload["baseline_evidence"]["summary_path"] = "../summary.json"  # type: ignore[index]

    with pytest.raises(ModelingContractError, match="safe repository-relative"):
        parse_candidate_config(json.dumps(payload))


def test_candidate_loader_rejects_missing_or_changed_references(tmp_path: Path) -> None:
    payload = _payload()
    payload["baseline_evidence"]["summary_path"] = "missing.json"  # type: ignore[index]
    config_path = tmp_path / "candidate.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelingContractError, match="Unable to read candidate baseline summary"):
        load_candidate_config(config_path, repo_root=tmp_path)


def test_candidate_loader_reports_a_missing_configuration(tmp_path: Path) -> None:
    with pytest.raises(ModelingContractError, match="Unable to load"):
        load_candidate_config(tmp_path / "missing.json", repo_root=tmp_path)


def test_candidate_loader_rejects_reference_hash_mismatch(tmp_path: Path) -> None:
    payload = _payload()
    summary_path = tmp_path / "summary.json"
    summary_path.write_bytes(b"{}\n")
    payload["baseline_evidence"]["summary_path"] = "summary.json"  # type: ignore[index]
    payload["baseline_evidence"]["summary_sha256"] = "0" * 64  # type: ignore[index]
    config_path = tmp_path / "candidate.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ModelingContractError, match="hash mismatch"):
        load_candidate_config(config_path, repo_root=tmp_path)


def test_checked_in_candidate_digest_matches_the_reviewed_protocol() -> None:
    content = CONFIG_PATH.read_bytes()
    assert hashlib.sha256(content).hexdigest() == (
        "556771afb87345a9ba54f5b1f7f60107a44c9d2a0b270a5fe66d1257ab89a695"
    )
    assert CandidateExperimentConfig.model_validate_json(content).status == "frozen_pre_experiment"
