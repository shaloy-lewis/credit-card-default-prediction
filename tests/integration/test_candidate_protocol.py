"""Integrity and semantic checks for the frozen Phase 3 candidate protocol."""

from __future__ import annotations

import hashlib
import json
import math
import tomllib
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "modeling" / "candidate_v1.json"

# Change this digest only after explicit review of the complete pre-experiment
# candidate protocol. Candidate results must not be consulted during that review.
EXPECTED_CONFIG_SHA256 = "93aa5331c4e558f6c4c1ce1fb9fce4ae16478a16567243fa6db723e031cf3f6c"
DEMOGRAPHIC_COLUMNS = {
    "sex_code",
    "education_code",
    "marital_status_code",
    "age_years",
}


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def test_candidate_protocol_is_frozen_bounded_and_holdout_blind() -> None:
    config_bytes = CONFIG_PATH.read_bytes()
    assert _sha256(config_bytes) == EXPECTED_CONFIG_SHA256
    config = json.loads(config_bytes)

    assert config["schema_version"] == "1.0.0"
    assert config["protocol_id"] == "candidate_v1"
    assert config["status"] == "frozen_pre_experiment"

    baseline = config["baseline_evidence"]
    summary_path = REPOSITORY_ROOT / baseline["summary_path"]
    report_path = REPOSITORY_ROOT / baseline["report_path"]
    summary_bytes = summary_path.read_bytes()
    assert _sha256(summary_bytes) == baseline["summary_sha256"]
    assert _sha256(report_path.read_bytes()) == baseline["report_sha256"]
    assert baseline["reference_model_id"] == "logistic_l2"
    baseline_summary = json.loads(summary_bytes)
    logistic = baseline_summary["models"][baseline["reference_model_id"]]["repeat_summaries"]
    reference = baseline["reference_metrics"]
    assert reference == {
        "average_precision_mean": logistic["average_precision"]["mean"],
        "average_precision_standard_deviation": logistic["average_precision"]["standard_deviation"],
        "brier_score_mean": logistic["brier_score"]["mean"],
        "lift_at_0_1_mean": logistic["capacity_0_1.lift"]["mean"],
        "roc_auc_mean": logistic["roc_auc"]["mean"],
    }

    data = config["data_contract"]
    assert data["partition"] == "development"
    assert data["expected_rows"] == 24_000
    assert data["holdout_access"] == "prohibited"
    assert data["cross_validation"] == {
        "n_splits": 5,
        "n_repeats": 3,
        "random_state": 42,
        "assignment_source": "sealed_phase_1_assignments",
    }
    assert data["split_assignment_sha256"] == baseline_summary["lineage"]["assignment_sha256"]

    feature_contract_path = REPOSITORY_ROOT / data["feature_contract_path"]
    feature_contract_bytes = feature_contract_path.read_bytes()
    assert _sha256(feature_contract_bytes) == data["feature_contract_sha256"]
    feature_contract = json.loads(feature_contract_bytes)

    views = {view["view_id"]: view for view in config["feature_views"]}
    assert set(views) == {
        "repayment_status_only",
        "monetary_only",
        "operational_full",
    }
    for view in views.values():
        predictors = view["predictor_columns"]
        assert len(predictors) == view["predictor_count"]
        assert len(predictors) == len(set(predictors))
        assert DEMOGRAPHIC_COLUMNS.isdisjoint(predictors)
        assert data["id_column"] not in predictors
        assert data["target_column"] not in predictors

    status_predictors = set(views["repayment_status_only"]["predictor_columns"])
    monetary_predictors = set(views["monetary_only"]["predictor_columns"])
    full_predictors = views["operational_full"]["predictor_columns"]
    assert status_predictors.isdisjoint(monetary_predictors)
    assert status_predictors | monetary_predictors == set(full_predictors)
    assert full_predictors == feature_contract["columns"]["predictor_columns"]

    candidate = config["candidate"]
    assert candidate["model_family"] == "catboost"
    assert candidate["library_version"] == "1.2.5"
    assert candidate["additional_challenger"] is None
    project = tomllib.loads((REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    assert f"catboost=={candidate['library_version']}" in dependencies
    handling = candidate["feature_handling"]
    assert set(handling["native_categorical_columns"]) == status_predictors
    assert set(handling["numeric_columns"]) == monetary_predictors
    assert handling["categorical_value_representation"] == "validated_integer_code_as_string"
    assert {
        handling["imputation"],
        handling["clipping"],
        handling["scaling"],
        handling["resampling"],
        handling["target_encoding"],
    } == {"none"}
    fixed = candidate["fixed_parameters"]
    assert fixed["loss_function"] == "Logloss"
    assert fixed["task_type"] == "CPU"
    assert fixed["bootstrap_type"] == "Bayesian"
    assert fixed["class_weights"] is None
    assert fixed["auto_class_weights"] is None
    assert fixed["random_seed"] == 42
    assert fixed["thread_count"] == 1
    assert fixed["allow_writing_files"] is False
    assert fixed["use_best_model"] is False
    assert fixed["early_stopping_rounds"] is None

    search = candidate["search"]
    assert search["strategy"] == "sklearn_parameter_sampler_without_replacement"
    assert search["sampler_library"] == "scikit-learn"
    assert search["sampler_version"] == "1.4.2"
    assert f"scikit-learn=={search['sampler_version']}" in dependencies
    assert search["random_state"] == 42
    assert search["n_iter"] == 12
    assert search["search_feature_view"] == "operational_full"
    assert search["evaluation_assignments"] == "all_5_folds_x_3_repeats"
    assert search["ablation_policy"]["hyperparameters"] == ("selected_from_operational_full_search")
    available_configurations = math.prod(
        len(values) for values in search["parameter_space"].values()
    )
    assert available_configurations == 324
    assert search["n_iter"] <= available_configurations
    folds = data["cross_validation"]["n_splits"] * data["cross_validation"]["n_repeats"]
    additional_ablation_views = len(search["ablation_policy"]["evaluation_feature_views"]) - 1
    expected_maximum_fits = search["n_iter"] * folds + additional_ablation_views * folds
    assert search["maximum_fold_fits"] == expected_maximum_fits == 210

    gate = config["advancement_gate"]
    relative = gate["relative_thresholds"]
    absolute = gate["derived_absolute_thresholds"]
    assert gate["combination_rule"] == "all_conditions_required"
    assert absolute["minimum_average_precision_mean"] == pytest.approx(
        reference["average_precision_mean"] + relative["minimum_average_precision_improvement"]
    )
    assert absolute["maximum_brier_score_mean"] == pytest.approx(
        reference["brier_score_mean"] + relative["maximum_brier_score_degradation"]
    )
    assert absolute["minimum_lift_at_0_1_mean"] == pytest.approx(
        reference["lift_at_0_1_mean"] - relative["maximum_lift_at_0_1_regression"]
    )
    assert absolute["maximum_average_precision_repeat_standard_deviation"] == 0.01
    assert gate["equivalence_band_average_precision"] == 0.002
    assert gate["search_selection"] == "best_eligible_operational_full_configuration"
    assert gate["final_variant_selection"] == (
        "best_eligible_feature_view_using_selected_hyperparameters"
    )
    assert gate["tie_break_order"] == [
        "fewer_predictors",
        "lower_depth",
        "fewer_iterations",
        "higher_l2_leaf_reg",
        "lower_learning_rate",
        "lower_random_strength",
        "lower_bagging_temperature",
    ]
    assert gate["fallback_model_id"] == "logistic_l2"

    governance = config["governance"]
    assert governance["demographic_predictors"] == "prohibited"
    assert governance["candidate_results_available_when_frozen"] is False
    assert "one_time_holdout_evaluation" in governance["deferred_to_phase_4"]
