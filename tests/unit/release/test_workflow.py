"""Tests for zero-computation Release A publication and verification."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import credit_risk.release.workflow as workflow
from credit_risk.modeling.tracking import GitEvidence
from credit_risk.release.contracts import load_release_config
from credit_risk.release.workflow import ReleaseWorkflowError
from tests.unit.release.helpers import copy_release_repository, fake_data_result


def _clean_git(root: Path) -> GitEvidence:
    return GitEvidence(
        commit_sha="a" * 40,
        dirty=False,
        diff_sha256="b" * 64,
        repository_root=root,
    )


def _build(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, output: str = "reports/releases/release_a_v1"
) -> tuple[Path, workflow.ReleaseWorkflowResult]:
    config_path = copy_release_repository(root)
    monkeypatch.setattr(workflow, "collect_git_evidence", lambda _path: _clean_git(root))
    monkeypatch.setattr(
        workflow,
        "_verify_offline_data",
        lambda **_kwargs: fake_data_result(root),
    )
    result = workflow.run_release_build(
        data_root=root / "unused-data",
        config_path=config_path,
        uncertainty_source=workflow.DEFAULT_UNCERTAINTY_SOURCE,
        output_root=output,
    )
    return config_path, result


def _loaded_sources(root: Path) -> tuple[object, dict[str, object], dict[str, object]]:
    config_path = copy_release_repository(root)
    config = load_release_config(config_path)
    sources = workflow._load_and_authenticate_sources(root, config)
    uncertainty = json.loads((root / workflow.DEFAULT_UNCERTAINTY_SOURCE).read_bytes())
    return config, sources, uncertainty


def test_build_publishes_authenticated_dossier_without_computation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, result = _build(tmp_path, monkeypatch)
    evidence = result.evidence_root

    assert result.status == "complete"
    assert {path.name for path in evidence.iterdir()} == set(workflow.RELEASE_OUTPUTS)
    source_uncertainty = tmp_path / workflow.DEFAULT_UNCERTAINTY_SOURCE
    assert (
        evidence / "validation-uncertainty.json"
    ).read_bytes() == source_uncertainty.read_bytes()
    summary = json.loads((evidence / "summary.json").read_bytes())
    assert summary["selection"]["fit_count"] == 4
    assert summary["selection"]["winner_refitted"] is False
    assert summary["calibration"]["method"] == "identity"
    assert summary["uncertainty"]["resamples"] == 500
    assert summary["final_test"]["permanently_consumed"] is True
    assert all(value == "passed" for value in summary["release_criteria"].values())
    assert not any(summary["claims"].values())
    assert summary["evidence_boundary"] == {
        "bootstrap_generated": False,
        "final_test_reevaluated": False,
        "full_dataset_integrity_verification_performed": True,
        "model_deserialized": False,
        "prediction_generated": False,
        "stress_evidence": "deferred_to_g4_release_b",
        "test_partition_selected": False,
        "test_predictions_loaded": False,
        "training_performed": False,
    }

    source_uncertainty.unlink()
    monkeypatch.setattr(
        workflow,
        "_verify_offline_data",
        lambda **_kwargs: pytest.fail("verify must not load runtime data"),
    )
    verified = workflow.verify_release_evidence(
        expected_manifest_sha256=result.evidence_manifest_sha256,
        config_path=config_path,
        evidence_root="reports/releases/release_a_v1",
    )
    assert verified.summary_sha256 == result.summary_sha256


def test_workflow_has_no_model_execution_surface() -> None:
    source = inspect.getsource(workflow)
    for prohibited in (
        "predict_proba",
        "load_selected_bundle",
        ".fit(",
        "run_final_test",
        "run_selection",
        "bootstrap_predictions",
    ):
        assert prohibited not in source


def test_verifier_rejects_tampering_even_when_manifest_is_self_updated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path, result = _build(tmp_path, monkeypatch)
    evidence = result.evidence_root
    summary_path = evidence / "summary.json"
    summary_path.write_bytes(summary_path.read_bytes() + b" ")

    with pytest.raises(ReleaseWorkflowError, match="artifact digest mismatch"):
        workflow.verify_release_evidence(
            expected_manifest_sha256=result.evidence_manifest_sha256,
            config_path=config_path,
            evidence_root="reports/releases/release_a_v1",
        )

    manifest_path = evidence / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"]["summary.json"]["sha256"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    manifest_path.write_bytes(workflow._json_bytes(manifest))
    with pytest.raises(ReleaseWorkflowError, match="manifest digest differs"):
        workflow.verify_release_evidence(
            expected_manifest_sha256=result.evidence_manifest_sha256,
            config_path=config_path,
            evidence_root="reports/releases/release_a_v1",
        )


@pytest.mark.parametrize(
    ("relative", "message"),
    [
        ("reports/releases", "child directory"),
        ("reports/governance/release_a_v1", "reports/releases"),
        ("../release_a_v1", "reports/releases"),
    ],
)
def test_build_rejects_unsafe_output_before_data_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative: str,
    message: str,
) -> None:
    config_path = copy_release_repository(tmp_path)
    monkeypatch.setattr(workflow, "collect_git_evidence", lambda _path: _clean_git(tmp_path))
    monkeypatch.setattr(
        workflow,
        "_verify_offline_data",
        lambda **_kwargs: pytest.fail("unsafe path must fail before data access"),
    )

    with pytest.raises(ReleaseWorkflowError, match=message):
        workflow.run_release_build(config_path=config_path, output_root=relative)


def test_build_rejects_dirty_existing_and_foreign_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = copy_release_repository(tmp_path)
    dirty = SimpleNamespace(repository_root=tmp_path, dirty=True, commit_sha="a" * 40)
    monkeypatch.setattr(workflow, "collect_git_evidence", lambda _path: dirty)
    with pytest.raises(ReleaseWorkflowError, match="clean committed worktree"):
        workflow.run_release_build(config_path=config_path)

    monkeypatch.setattr(workflow, "collect_git_evidence", lambda _path: _clean_git(tmp_path))
    existing = tmp_path / "reports/releases/release_a_v1"
    existing.mkdir(parents=True)
    with pytest.raises(ReleaseWorkflowError, match="already exists"):
        workflow.run_release_build(config_path=config_path)
    existing.rmdir()

    with pytest.raises(ReleaseWorkflowError, match="exact reviewed"):
        workflow.run_release_build(
            config_path=config_path,
            uncertainty_source="experiment/foreign.json",
        )


def test_build_rejects_missing_corrupt_and_incompatible_uncertainty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = copy_release_repository(tmp_path)
    monkeypatch.setattr(workflow, "collect_git_evidence", lambda _path: _clean_git(tmp_path))
    monkeypatch.setattr(
        workflow,
        "_verify_offline_data",
        lambda **_kwargs: fake_data_result(tmp_path),
    )
    uncertainty = tmp_path / workflow.DEFAULT_UNCERTAINTY_SOURCE
    original = uncertainty.read_bytes()
    uncertainty.unlink()
    with pytest.raises(ReleaseWorkflowError, match="Unable to read validation uncertainty"):
        workflow.run_release_build(config_path=config_path)

    uncertainty.write_bytes(b"not json")
    with pytest.raises(ReleaseWorkflowError, match="digest differs"):
        workflow.run_release_build(config_path=config_path)

    uncertainty.write_bytes(original)
    payload = json.loads(original)
    payload["metrics"]["average_precision"]["lower"] = 0.9
    altered = workflow._json_bytes(payload)
    uncertainty.write_bytes(altered)
    config = json.loads(config_path.read_bytes())
    config["uncertainty_source"]["sha256"] = hashlib.sha256(altered).hexdigest()
    config_path.write_bytes(workflow._json_bytes(config))
    with pytest.raises(ReleaseWorkflowError, match="different uncertainty evidence"):
        workflow.run_release_build(config_path=config_path)


def test_source_and_data_lineage_mismatches_fail_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = copy_release_repository(tmp_path)
    monkeypatch.setattr(workflow, "collect_git_evidence", lambda _path: _clean_git(tmp_path))
    baseline = tmp_path / "reports/modeling/baseline_v1/summary.json"
    baseline.write_bytes(baseline.read_bytes() + b" ")
    with pytest.raises(ReleaseWorkflowError, match="source digest mismatch"):
        workflow.run_release_build(config_path=config_path)

    tmp_path_2 = tmp_path / "second"
    config_path = copy_release_repository(tmp_path_2)
    monkeypatch.setattr(workflow, "collect_git_evidence", lambda _path: _clean_git(tmp_path_2))
    bad = fake_data_result(tmp_path_2)
    bad.canonical_sha256 = "0" * 64
    monkeypatch.setattr(workflow, "_verify_offline_data", lambda **_kwargs: bad)
    with pytest.raises(ReleaseWorkflowError, match="Offline data verification differs"):
        workflow.run_release_build(config_path=config_path)


def test_atomic_publication_failure_leaves_no_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = copy_release_repository(tmp_path)
    monkeypatch.setattr(workflow, "collect_git_evidence", lambda _path: _clean_git(tmp_path))
    monkeypatch.setattr(
        workflow,
        "_verify_offline_data",
        lambda **_kwargs: fake_data_result(tmp_path),
    )
    monkeypatch.setattr(workflow.os, "replace", lambda *_args: (_ for _ in ()).throw(OSError("x")))

    with pytest.raises(ReleaseWorkflowError, match="atomically"):
        workflow.run_release_build(config_path=config_path)
    assert not (tmp_path / "reports/releases/release_a_v1").exists()


def test_staged_validation_failure_leaves_no_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = copy_release_repository(tmp_path)
    monkeypatch.setattr(workflow, "collect_git_evidence", lambda _path: _clean_git(tmp_path))
    monkeypatch.setattr(
        workflow,
        "_verify_offline_data",
        lambda **_kwargs: fake_data_result(tmp_path),
    )
    monkeypatch.setattr(
        workflow,
        "_validate_release_directory",
        lambda **_kwargs: (_ for _ in ()).throw(ReleaseWorkflowError("staged evidence rejected")),
    )

    with pytest.raises(ReleaseWorkflowError, match="staged evidence rejected"):
        workflow.run_release_build(config_path=config_path)
    assert not (tmp_path / "reports/releases/release_a_v1").exists()


def test_verifier_rejects_invalid_digest_missing_directory_and_extra_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_path = copy_release_repository(tmp_path)
    with pytest.raises(ReleaseWorkflowError, match="lowercase hexadecimal"):
        workflow.verify_release_evidence(
            expected_manifest_sha256="invalid",
            config_path=config_path,
        )
    with pytest.raises(ReleaseWorkflowError, match="directory is missing"):
        workflow.verify_release_evidence(
            expected_manifest_sha256="a" * 64,
            config_path=config_path,
        )

    config_path, result = _build(tmp_path, monkeypatch)
    (result.evidence_root / "unexpected").mkdir()
    with pytest.raises(ReleaseWorkflowError, match="differs from the allowlist"):
        workflow.verify_release_evidence(
            expected_manifest_sha256=result.evidence_manifest_sha256,
            config_path=config_path,
            evidence_root="reports/releases/release_a_v1",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda sources: sources["baseline_summary"].update({"schema_version": "2.0"}),
            "baseline summary",
        ),
        (
            lambda sources: sources["baseline_summary"]["experiment"].update(
                {"baseline_names": []}
            ),
            "baseline models",
        ),
        (
            lambda sources: sources["baseline_summary"]["data"].update({"holdout_evaluated": True}),
            "baseline evidence misstates holdout",
        ),
        (
            lambda sources: sources["baseline_summary"]["reproducibility"].update(
                {"git_dirty": True}
            ),
            "not produced cleanly",
        ),
        (
            lambda sources: sources["selection_summary"]["protocol"].update({"fit_count": 5}),
            "Selection protocol",
        ),
        (
            lambda sources: sources["selection_summary"]["models"][0].update({"model_id": "other"}),
            "model order",
        ),
        (
            lambda sources: sources["selection_summary"]["selection"].update(
                {"selected_model_id": "logistic_l2"}
            ),
            "reviewed winner",
        ),
        (
            lambda sources: sources["selection_summary"]["population"].update(
                {"validation_rows": 4799}
            ),
            "population differs",
        ),
        (
            lambda sources: sources["selection_summary"]["holdout"].update({"evaluated": True}),
            "misstates holdout evaluation",
        ),
        (
            lambda sources: sources["selection_summary"]["selected_model"][
                "calibration_diagnostics"
            ].update({"method": "isotonic"}),
            "calibration evidence",
        ),
        (
            lambda sources: sources["bundle_manifest"].update({"selected_model_id": "logistic_l2"}),
            "bundle manifest",
        ),
        (
            lambda sources: sources["final_test_authorization"].update({"status": "reusable"}),
            "authorization differs",
        ),
        (
            lambda sources: sources["final_test_approval"].update({"maximum_evaluations": 2}),
            "approval chain",
        ),
        (
            lambda sources: sources["final_test_started_receipt"].update({"status": "pending"}),
            "durable receipts",
        ),
        (
            lambda sources: sources["final_test_summary"].update({"g2_status": "open"}),
            "does not satisfy",
        ),
        (
            lambda sources: sources["final_test_summary"]["metrics"].update({"capacities": []}),
            "Final test capacities",
        ),
        (
            lambda sources: sources["selection_summary"]["reproducibility"]["data_lineage"].update(
                {"canonical_sha256": "0" * 64}
            ),
            "data lineage differs",
        ),
    ],
)
def test_source_semantics_reject_each_governed_mismatch(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    config, original_sources, uncertainty = _loaded_sources(tmp_path)
    sources = copy.deepcopy(original_sources)
    mutation(sources)  # type: ignore[operator]

    with pytest.raises(ReleaseWorkflowError, match=message):
        workflow._validate_source_semantics(config, sources, uncertainty)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"resamples": 499}), "metadata differs"),
        (lambda value: value.update({"metrics": {}}), "metrics differ"),
        (
            lambda value: value["metrics"]["average_precision"].update({"extra": 1}),
            "interval is malformed",
        ),
        (
            lambda value: value["metrics"]["average_precision"].update({"point": 0.5}),
            "point differs",
        ),
        (
            lambda value: value["metrics"]["average_precision"].update({"lower": 0.9}),
            "interval is unordered",
        ),
    ],
)
def test_uncertainty_validation_rejects_altered_contract(
    tmp_path: Path, mutation: object, message: str
) -> None:
    config, sources, original = _loaded_sources(tmp_path)
    uncertainty = copy.deepcopy(original)
    mutation(uncertainty)  # type: ignore[operator]
    selected_metrics = sources["selection_summary"]["models"][-1]["validation_metrics"]  # type: ignore[index]

    with pytest.raises(ReleaseWorkflowError, match=message):
        workflow._validate_uncertainty(config, uncertainty, selected_metrics)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("items", "message"),
    [
        (None, "capacities differ"),
        ([{"capacity": 0.05}], "capacities differ"),
        (
            [
                {
                    "capacity": capacity,
                    "selected_count": 0,
                    "precision": 0.1,
                    "recall": 0.1,
                    "lift": 1.0,
                    "expected_true_positives": 1.0,
                }
                for capacity in workflow.CAPACITIES
            ],
            "wrong selected count",
        ),
    ],
)
def test_capacity_validation_rejects_incomplete_evidence(items: object, message: str) -> None:
    with pytest.raises(ReleaseWorkflowError, match=message):
        workflow._validate_capacities(items, 100, "validation")


def test_capacity_validation_rejects_missing_metric() -> None:
    items = [
        {
            "capacity": capacity,
            "selected_count": int(100 * capacity),
            "precision": 0.1,
            "recall": 0.1,
            "lift": 1.0,
            "expected_true_positives": 1.0,
        }
        for capacity in workflow.CAPACITIES
    ]
    items[-1].pop("lift")
    with pytest.raises(ReleaseWorkflowError, match="missing lift"):
        workflow._validate_capacities(items, 100, "validation")


def _write_manifest_and_digest(path: Path, manifest: dict[str, object]) -> str:
    path.write_bytes(workflow._json_bytes(manifest))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update({"release_id": "other"}), "manifest is incompatible"),
        (lambda value: value.update({"source_artifacts": {}}), "source bindings differ"),
        (lambda value: value.update({"artifacts": {}}), "artifact allowlist is incomplete"),
        (
            lambda value: value["artifacts"].update({"summary.json": []}),
            "verification failed",
        ),
        (
            lambda value: value["artifacts"]["summary.json"].update({"row_level_data": True}),
            "unsafe row-data claim",
        ),
        (lambda value: value.update({"boundaries_verified": {}}), "misstates the evidence"),
    ],
)
def test_verifier_rejects_semantically_altered_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: object,
    message: str,
) -> None:
    config_path, result = _build(tmp_path, monkeypatch)
    manifest_path = result.evidence_root / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    mutation(manifest)  # type: ignore[operator]
    expected = _write_manifest_and_digest(manifest_path, manifest)

    with pytest.raises(ReleaseWorkflowError, match=message):
        workflow.verify_release_evidence(
            expected_manifest_sha256=expected,
            config_path=config_path,
            evidence_root="reports/releases/release_a_v1",
        )


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        (None, "status", "draft", "identity or status"),
        ("lineage", "release_config_sha256", "0" * 64, "configuration lineage"),
        ("lineage", "implementation_git_commit", "0" * 40, "implementation lineage"),
        ("lineage", "git_dirty", True, "clean implementation"),
        (None, "release_criteria", {}, "pass every frozen criterion"),
        ("selection", "selected_model_id", "other", "wrong selected model"),
        ("final_test", "g2_status", "open", "closed G2"),
        ("evidence_boundary", "stress_evidence", "complete", "stress-evidence boundary"),
        ("claims", "production_suitability", True, "unsupported positive claim"),
        ("data", "assignment_sha256", "0" * 64, "data lineage is inconsistent"),
    ],
)
def test_verifier_rejects_semantically_altered_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    section: str | None,
    key: str,
    value: object,
    message: str,
) -> None:
    config_path, result = _build(tmp_path, monkeypatch)
    summary_path = result.evidence_root / "summary.json"
    summary = json.loads(summary_path.read_bytes())
    target = summary if section is None else summary[section]
    target[key] = value
    summary_path.write_bytes(workflow._json_bytes(summary))
    manifest_path = result.evidence_root / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["artifacts"]["summary.json"]["sha256"] = hashlib.sha256(
        summary_path.read_bytes()
    ).hexdigest()
    expected = _write_manifest_and_digest(manifest_path, manifest)

    with pytest.raises(ReleaseWorkflowError, match=message):
        workflow.verify_release_evidence(
            expected_manifest_sha256=expected,
            config_path=config_path,
            evidence_root="reports/releases/release_a_v1",
        )


def test_low_level_helpers_fail_safely(tmp_path: Path) -> None:
    with pytest.raises(ReleaseWorkflowError, match="payload differs"):
        workflow._publish_directory_atomically(tmp_path / "release", {})
    with pytest.raises(ReleaseWorkflowError, match="must contain a JSON object"):
        workflow._read_json_bytes(b"[]", "fixture")
    with pytest.raises(ReleaseWorkflowError, match="Unable to parse"):
        workflow._read_json_bytes(b"not-json", "fixture")
    with pytest.raises(ReleaseWorkflowError, match="Unable to read fixture"):
        workflow._read_text(tmp_path / "missing", "fixture")
    with pytest.raises(ReleaseWorkflowError, match="Unable to hash"):
        workflow._sha256_file(tmp_path / "missing")
    with pytest.raises(ReleaseWorkflowError, match="Unable to locate repository"):
        workflow._repository_root(tmp_path / "config.json")
