"""Tests for the narrow SQLite MLflow tracking adapter."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from contextlib import AbstractContextManager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from credit_risk.modeling import tracking
from credit_risk.modeling.tracking import (
    BASELINE_MODEL_NAMES,
    ModelRunPayload,
    TrackingDependencyError,
    TrackingError,
    collect_git_evidence,
    collect_package_versions,
    mark_tracking_run_failed,
    track_baseline_runs,
    track_candidate_runs,
)


class _RunContext(AbstractContextManager[SimpleNamespace]):
    def __init__(self, mlflow: _FakeMlflow, run_id: str) -> None:
        self.mlflow = mlflow
        self.run_id = run_id

    def __enter__(self) -> SimpleNamespace:
        self.mlflow.active_runs.append(self.run_id)
        return SimpleNamespace(info=SimpleNamespace(run_id=self.run_id))

    def __exit__(self, *_args: object) -> None:
        assert self.mlflow.active_runs.pop() == self.run_id


class _FakeClient:
    def __init__(self) -> None:
        self.experiment: SimpleNamespace | None = None
        self.created: list[tuple[str, str]] = []
        self.tracking_uris: list[str] = []
        self.tags: list[tuple[str, str, str]] = []
        self.terminations: list[tuple[str, str]] = []

    def factory(self, *, tracking_uri: str) -> _FakeClient:
        self.tracking_uris.append(tracking_uri)
        return self

    def get_experiment_by_name(self, _name: str) -> SimpleNamespace | None:
        return self.experiment

    def create_experiment(self, name: str, *, artifact_location: str) -> str:
        self.created.append((name, artifact_location))
        return "experiment-1"

    def set_tag(self, run_id: str, key: str, value: str) -> None:
        self.tags.append((run_id, key, value))

    def set_terminated(self, run_id: str, *, status: str) -> None:
        self.terminations.append((run_id, status))


class _FakeMlflow:
    def __init__(self) -> None:
        self.client = _FakeClient()
        self.tracking = SimpleNamespace(MlflowClient=self.client.factory)
        self.tracking_uri = "sqlite:///prior/mlflow.db"
        self.tracking_uri_history: list[str] = []
        self.started: list[dict[str, Any]] = []
        self.active_runs: list[str] = []
        self.params: list[tuple[str, dict[str, object]]] = []
        self.metrics: list[tuple[str, dict[str, float]]] = []
        self.artifacts: list[tuple[str, Path, str]] = []

    def set_tracking_uri(self, uri: str) -> None:
        self.tracking_uri = uri
        self.tracking_uri_history.append(uri)

    def get_tracking_uri(self) -> str:
        return self.tracking_uri

    def start_run(self, **kwargs: object) -> _RunContext:
        run_id = f"run-{len(self.started)}"
        self.started.append({"run_id": run_id, **kwargs})
        return _RunContext(self, run_id)

    def log_params(self, values: dict[str, object]) -> None:
        self.params.append((self.active_runs[-1], values))

    def log_metrics(self, values: dict[str, float]) -> None:
        self.metrics.append((self.active_runs[-1], values))

    def log_artifact(self, path: str, *, artifact_path: str) -> None:
        self.artifacts.append((self.active_runs[-1], Path(path), artifact_path))


def test_tracks_one_parent_and_three_nested_children_with_allowlisted_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMlflow()
    monkeypatch.setattr(tracking, "_load_mlflow", lambda: fake)
    artifacts = _evidence_files(tmp_path)
    payloads = tuple(
        ModelRunPayload(model_name=name, parameters={"kind": name}, metrics={"score": 0.5})
        for name in BASELINE_MODEL_NAMES
    )

    result = track_baseline_runs(
        tracking_root=tmp_path / "tracking",
        experiment_name="credit-risk-baseline-v1",
        parent_run_name="baseline_v1",
        parent_parameters={"dirty": False, "seed": 42},
        parent_tags={"partition": "development"},
        model_runs=payloads,
        artifacts=artifacts,
    )

    assert result.parent_run_id == "run-0"
    assert result.child_run_ids == tuple(
        (name, f"run-{index}") for index, name in enumerate(BASELINE_MODEL_NAMES, start=1)
    )
    assert fake.tracking_uri == "sqlite:///prior/mlflow.db"
    assert fake.tracking_uri_history == [result.tracking_uri, "sqlite:///prior/mlflow.db"]
    assert result.tracking_uri.endswith("/tracking/mlflow.db")
    assert fake.client.created == [
        (
            "credit-risk-baseline-v1",
            (tmp_path / "tracking" / "artifacts").resolve().as_uri(),
        )
    ]
    assert len(fake.started) == 4
    assert "nested" not in fake.started[0]
    assert [call["run_name"] for call in fake.started[1:]] == list(BASELINE_MODEL_NAMES)
    assert all(call["nested"] is True for call in fake.started[1:])
    assert [record[0] for record in fake.params] == ["run-0", "run-1", "run-2", "run-3"]
    assert [record[0] for record in fake.metrics] == ["run-1", "run-2", "run-3"]
    assert [(path.name, destination) for _, path, destination in fake.artifacts] == [
        ("summary.json", "evidence"),
        ("baseline-report.md", "evidence"),
        ("oof_predictions.csv", "runtime"),
        ("logistic_fold_diagnostics.json", "runtime"),
    ]


def test_reuses_only_an_experiment_with_the_expected_artifact_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMlflow()
    fake.client.experiment = SimpleNamespace(
        experiment_id="existing",
        artifact_location="file:///unexpected",
    )
    monkeypatch.setattr(tracking, "_load_mlflow", lambda: fake)

    with pytest.raises(TrackingError, match="unexpected artifact location"):
        track_baseline_runs(
            tracking_root=tmp_path / "tracking",
            experiment_name="credit-risk-baseline-v1",
            parent_run_name="baseline_v1",
            parent_parameters={},
            parent_tags={},
            model_runs=_payloads(),
            artifacts=_evidence_files(tmp_path),
        )

    assert fake.started == []


def test_marks_parent_failed_after_post_tracking_publication_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMlflow()
    monkeypatch.setattr(tracking, "_load_mlflow", lambda: fake)
    result = tracking.TrackingRunResult(
        tracking_uri="sqlite:///tracking/mlflow.db",
        experiment_name="credit-risk-baseline-v1",
        parent_run_id="parent-run",
        child_run_ids=(),
    )

    mark_tracking_run_failed(result, failure_stage="evidence_publication")

    assert fake.client.tags == [("parent-run", "workflow_failure_stage", "evidence_publication")]
    assert fake.client.terminations == [("parent-run", "FAILED")]

    with pytest.raises(TrackingError, match="must not be blank"):
        mark_tracking_run_failed(result, failure_stage=" ")


@pytest.mark.integration
def test_real_sqlite_store_records_parent_child_hierarchy_and_only_safe_artifacts(
    tmp_path: Path,
) -> None:
    import mlflow

    result = track_baseline_runs(
        tracking_root=tmp_path / "tracking",
        experiment_name="test-credit-risk-baseline",
        parent_run_name="baseline_v1",
        parent_parameters={"config_sha256": "a" * 64},
        parent_tags={"partition": "development"},
        model_runs=_payloads(),
        artifacts=_evidence_files(tmp_path),
    )

    client = mlflow.tracking.MlflowClient(tracking_uri=result.tracking_uri)
    experiment = client.get_experiment_by_name("test-credit-risk-baseline")
    assert experiment is not None
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 4
    children = [run for run in runs if run.data.tags.get("mlflow.parentRunId")]
    assert len(children) == 3
    assert {run.data.tags["model_id"] for run in children} == set(BASELINE_MODEL_NAMES)
    assert all(run.data.tags["mlflow.parentRunId"] == result.parent_run_id for run in children)
    root_artifacts = client.list_artifacts(result.parent_run_id)
    assert {artifact.path for artifact in root_artifacts} == {"evidence", "runtime"}
    artifact_names = {
        artifact.path
        for directory in ("evidence", "runtime")
        for artifact in client.list_artifacts(result.parent_run_id, directory)
    }
    assert artifact_names == {
        "evidence/summary.json",
        "evidence/baseline-report.md",
        "runtime/oof_predictions.csv",
        "runtime/logistic_fold_diagnostics.json",
    }
    assert (tmp_path / "tracking" / "mlflow.db").is_file()


def test_rejects_model_or_artifact_scope_before_mlflow_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        tracking,
        "_load_mlflow",
        lambda: (_ for _ in ()).throw(AssertionError("MLflow should not load")),
    )
    invalid_models = (ModelRunPayload(model_name="logistic_l2", parameters={}, metrics={}),)
    with pytest.raises(TrackingError, match="exactly the ordered model runs"):
        track_baseline_runs(
            tracking_root=tmp_path,
            experiment_name="experiment",
            parent_run_name="parent",
            parent_parameters={},
            parent_tags={},
            model_runs=invalid_models,
            artifacts=_evidence_files(tmp_path),
        )

    unsafe = tmp_path / "model.pkl"
    unsafe.write_bytes(b"pickle")
    with pytest.raises(TrackingError, match="allowlist mismatch"):
        track_baseline_runs(
            tracking_root=tmp_path,
            experiment_name="experiment",
            parent_run_name="parent",
            parent_parameters={},
            parent_tags={},
            model_runs=_payloads(),
            artifacts=(*_evidence_files(tmp_path), unsafe),
        )

    summary, report, oof, diagnostics = _evidence_files(tmp_path)
    oof.write_text("ID,X1,Y\n1,10000,0\n", encoding="utf-8", newline="\n")
    summary_payload = json.loads(summary.read_text(encoding="utf-8"))
    summary_payload["runtime_artifacts"]["oof_predictions_sha256"] = hashlib.sha256(
        oof.read_bytes()
    ).hexdigest()
    summary_bytes = (json.dumps(summary_payload, sort_keys=True) + "\n").encode("utf-8")
    summary.write_bytes(summary_bytes)
    report.write_text(
        "# Governed baseline experiment report\n\n"
        "**Deterministic summary SHA-256:** "
        f"`{hashlib.sha256(summary_bytes).hexdigest()}`\n",
        encoding="utf-8",
        newline="\n",
    )
    with pytest.raises(TrackingError, match="unexpected schema"):
        track_baseline_runs(
            tracking_root=tmp_path,
            experiment_name="experiment",
            parent_run_name="parent",
            parent_parameters={},
            parent_tags={},
            model_runs=_payloads(),
            artifacts=(summary, report, oof, diagnostics),
        )


def test_git_evidence_hashes_tracked_diff_and_untracked_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    untracked = tmp_path / "new.py"
    untracked.write_text("first\n", encoding="utf-8")

    def fake_git(_root: Path, *arguments: str) -> bytes:
        if arguments[:2] == ("rev-parse", "--show-toplevel"):
            return f"{tmp_path}\n".encode()
        if arguments[:2] == ("rev-parse", "HEAD"):
            return ("a" * 40 + "\n").encode()
        if arguments[0] == "status":
            return b" M tracked.py\0?? new.py\0"
        if arguments[0] == "diff":
            return b"tracked binary diff"
        raise AssertionError(arguments)

    monkeypatch.setattr(tracking, "_run_git", fake_git)
    first = collect_git_evidence(tmp_path)
    untracked.write_text("second\n", encoding="utf-8")
    second = collect_git_evidence(tmp_path)

    assert first.commit_sha == "a" * 40
    assert first.dirty
    assert first.diff_sha256 != second.diff_sha256
    assert "first" not in first.diff_sha256


def test_git_evidence_rejects_paths_outside_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_git(_root: Path, *arguments: str) -> bytes:
        if arguments[:2] == ("rev-parse", "--show-toplevel"):
            return f"{tmp_path}\n".encode()
        if arguments[:2] == ("rev-parse", "HEAD"):
            return ("b" * 40 + "\n").encode()
        if arguments[0] == "status":
            return b"?? ../outside.txt\0"
        return b""

    monkeypatch.setattr(tracking, "_run_git", fake_git)
    with pytest.raises(TrackingError, match="outside the repository"):
        collect_git_evidence(tmp_path)


def test_package_versions_are_sorted_and_missing_metadata_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(tracking.importlib.metadata, "version", lambda name: f"1.0-{name}")
    assert list(collect_package_versions(("zeta", "alpha", "alpha"))) == ["alpha", "zeta"]

    def missing(_name: str) -> str:
        raise tracking.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(tracking.importlib.metadata, "version", missing)
    with pytest.raises(TrackingDependencyError, match="modeling.*extra"):
        collect_package_versions(("mlflow",))
    with pytest.raises(TrackingDependencyError, match="data.*extra"):
        collect_package_versions(("pandera",))


def test_rejects_oof_rows_with_model_specific_fold_assignments(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, report, oof, diagnostics = _evidence_files(tmp_path)
    reader = csv.DictReader(io.StringIO(oof.read_text(encoding="utf-8"), newline=""))
    rows = list(reader)
    fieldnames = tuple(reader.fieldnames or ())
    logistic_rows = {
        int(row["account_id"]): row
        for row in rows
        if row["model_id"] == "logistic_l2" and row["repeat_index"] == "0"
    }
    logistic_rows[1]["fold_index"], logistic_rows[2]["fold_index"] = (
        logistic_rows[2]["fold_index"],
        logistic_rows[1]["fold_index"],
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    oof.write_bytes(output.getvalue().encode("utf-8"))
    _rebind_evidence(summary, report, oof, diagnostics)
    monkeypatch.setattr(
        tracking,
        "_load_mlflow",
        lambda: (_ for _ in ()).throw(AssertionError("MLflow should not load")),
    )

    with pytest.raises(TrackingError, match="inconsistent fold assignments"):
        track_baseline_runs(
            tracking_root=tmp_path / "tracking",
            experiment_name="experiment",
            parent_run_name="parent",
            parent_parameters={},
            parent_tags={},
            model_runs=_payloads(),
            artifacts=(summary, report, oof, diagnostics),
        )


def test_rejects_string_instead_of_logistic_feature_name_array(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    summary, report, oof, diagnostics = _evidence_files(tmp_path)
    payload = json.loads(diagnostics.read_text(encoding="utf-8"))
    payload["transformed_feature_names"] = "abc"
    for fold in payload["folds"]:
        fold["coefficients"] = [0.1, 0.2, 0.3]
    diagnostics.write_bytes((json.dumps(payload, sort_keys=True) + "\n").encode("utf-8"))
    _rebind_evidence(summary, report, oof, diagnostics)
    monkeypatch.setattr(
        tracking,
        "_load_mlflow",
        lambda: (_ for _ in ()).throw(AssertionError("MLflow should not load")),
    )

    with pytest.raises(TrackingError, match="fold evidence contract"):
        track_baseline_runs(
            tracking_root=tmp_path / "tracking",
            experiment_name="experiment",
            parent_run_name="parent",
            parent_parameters={},
            parent_tags={},
            model_runs=_payloads(),
            artifacts=(summary, report, oof, diagnostics),
        )


def test_tracks_candidate_parent_and_fourteen_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = _FakeMlflow()
    monkeypatch.setattr(tracking, "_load_mlflow", lambda: fake)
    artifacts = tuple(
        tmp_path / name
        for name in (
            "summary.json",
            "candidate-report.md",
            "oof_predictions.csv",
            "fold_diagnostics.json",
        )
    )
    monkeypatch.setattr(
        tracking,
        "_validate_candidate_artifacts",
        lambda _paths: tuple(path.resolve() for path in artifacts),
    )
    names = tuple(f"operational_full__cb_cfg_{index:03d}" for index in range(1, 13)) + (
        "repayment_status_only__cb_cfg_004",
        "monetary_only__cb_cfg_004",
    )
    payloads = tuple(
        ModelRunPayload(model_name=name, parameters={"seed": 42}, metrics={"ap": 0.5})
        for name in names
    )

    result = track_candidate_runs(
        tracking_root=tmp_path / "tracking",
        parent_parameters={"dirty": False},
        parent_tags={"partition": "development"},
        variant_runs=payloads,
        artifacts=artifacts,
    )

    assert result.experiment_name == "credit-risk-candidate-v1"
    assert result.parent_run_id == "run-0"
    assert result.child_run_ids == tuple(
        (name, f"run-{index}") for index, name in enumerate(names, start=1)
    )
    assert len(fake.started) == 15
    assert [call["run_name"] for call in fake.started[1:]] == list(names)
    assert [(path.name, destination) for _, path, destination in fake.artifacts] == [
        ("summary.json", "evidence"),
        ("candidate-report.md", "evidence"),
        ("oof_predictions.csv", "runtime"),
        ("fold_diagnostics.json", "runtime"),
    ]


def test_candidate_tracking_rejects_incomplete_variant_payloads(tmp_path: Path) -> None:
    with pytest.raises(TrackingError, match="14 unique"):
        track_candidate_runs(
            tracking_root=tmp_path,
            parent_parameters={},
            parent_tags={},
            variant_runs=(),
            artifacts=(),
        )


def test_candidate_artifact_boundary_accepts_only_hash_bound_non_executable_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _candidate_evidence_files(tmp_path)
    monkeypatch.setattr(tracking, "_validate_candidate_oof", lambda *_args: None)

    validated = tracking._validate_candidate_artifacts(artifacts)

    assert {path.name for path in validated} == set(tracking.CANDIDATE_TRACKING_ARTIFACTS)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("data", "development-only"),
        ("fits", "210 fold fits"),
        ("boundary", "artifact boundary"),
        ("oof_hash", "OOF hashes"),
        ("diagnostics_hash", "diagnostics hashes"),
        ("diagnostics_count", "exactly 210"),
        ("report", "not bound"),
        ("forbidden", "operational identifiers"),
    ),
)
def test_candidate_artifact_boundary_rejects_governance_drift(
    mutation: str,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _candidate_evidence_files(tmp_path)
    by_name = {path.name: path for path in artifacts}
    summary = json.loads(by_name["summary.json"].read_text(encoding="utf-8"))
    diagnostics = json.loads(by_name["fold_diagnostics.json"].read_text(encoding="utf-8"))
    if mutation == "data":
        summary["data"]["holdout_evaluated"] = True
    elif mutation == "fits":
        summary["fit_budget"]["completed_fold_fits"] = 209
    elif mutation == "boundary":
        summary["runtime_artifacts"]["contains_fitted_models"] = True
    elif mutation == "oof_hash":
        summary["runtime_artifacts"]["oof_predictions_sha256"] = "0" * 64
    elif mutation == "diagnostics_hash":
        summary["runtime_artifacts"]["fold_diagnostics_sha256"] = "0" * 64
    elif mutation == "diagnostics_count":
        diagnostics["fit_count"] = 209
        by_name["fold_diagnostics.json"].write_text(json.dumps(diagnostics), encoding="utf-8")
        summary["runtime_artifacts"]["fold_diagnostics_sha256"] = hashlib.sha256(
            by_name["fold_diagnostics.json"].read_bytes()
        ).hexdigest()
    elif mutation == "report":
        by_name["candidate-report.md"].write_text("wrong", encoding="utf-8")
    elif mutation == "forbidden":
        summary["tracking_uri"] = "sqlite:///forbidden"
    summary_bytes = json.dumps(summary, sort_keys=True).encode("utf-8")
    by_name["summary.json"].write_bytes(summary_bytes)
    if mutation != "report":
        by_name["candidate-report.md"].write_text(
            "# Governed CatBoost candidate report\n\n"
            f"Deterministic summary SHA-256:** `{hashlib.sha256(summary_bytes).hexdigest()}`\n",
            encoding="utf-8",
            newline="\n",
        )
    monkeypatch.setattr(tracking, "_validate_candidate_oof", lambda *_args: None)

    with pytest.raises(TrackingError, match=message):
        tracking._validate_candidate_artifacts(artifacts)


def test_candidate_artifact_allowlist_and_payload_roles_are_strict(tmp_path: Path) -> None:
    artifacts = _candidate_evidence_files(tmp_path)
    with pytest.raises(TrackingError, match="unique"):
        tracking._validate_candidate_artifacts((*artifacts, artifacts[0]))
    with pytest.raises(TrackingError, match="allowlist"):
        tracking._validate_candidate_artifacts(artifacts[:-1])
    artifacts[-1].unlink()
    with pytest.raises(TrackingError, match="missing or not a file"):
        tracking._validate_candidate_artifacts(artifacts)

    wrong_search = tuple(
        ModelRunPayload(model_name=f"wrong-{index}", parameters={}, metrics={})
        for index in range(14)
    )
    with pytest.raises(TrackingError, match="ordered cb_cfg"):
        tracking._validate_candidate_payloads(wrong_search)
    wrong_diagnostics = tuple(
        ModelRunPayload(
            model_name=f"operational_full__cb_cfg_{index:03d}", parameters={}, metrics={}
        )
        for index in range(1, 13)
    ) + (
        ModelRunPayload(model_name="wrong-a", parameters={}, metrics={}),
        ModelRunPayload(model_name="wrong-b", parameters={}, metrics={}),
    )
    with pytest.raises(TrackingError, match="diagnostic variants"):
        tracking._validate_candidate_payloads(wrong_diagnostics)

    mismatched_diagnostics = wrong_diagnostics[:12] + (
        ModelRunPayload(model_name="repayment_status_only__cb_cfg_001", parameters={}, metrics={}),
        ModelRunPayload(model_name="monetary_only__cb_cfg_002", parameters={}, metrics={}),
    )
    with pytest.raises(TrackingError, match="reuse one reviewed"):
        tracking._validate_candidate_payloads(mismatched_diagnostics)


@pytest.mark.parametrize(
    ("row", "message"),
    (
        ({"probability": "nan", "repeat_index": "0", "fold_index": "0"}, "probability"),
        ({"probability": "0.5", "repeat_index": "3", "fold_index": "0"}, "fold"),
    ),
)
def test_candidate_oof_validator_rejects_invalid_rows(
    row: dict[str, str],
    message: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "oof_predictions.csv"
    fields = [
        "account_id",
        "variant_id",
        "feature_view",
        "configuration_id",
        "repeat_index",
        "fold_index",
        "probability",
    ]
    complete = {
        "account_id": "1",
        "variant_id": "variant",
        "feature_view": "operational_full",
        "configuration_id": "cb_cfg_001",
        **row,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerow(complete)

    with pytest.raises(TrackingError, match=message):
        tracking._validate_candidate_oof(path, {"oof_prediction_rows": 1})


def _payloads() -> tuple[ModelRunPayload, ...]:
    return tuple(
        ModelRunPayload(model_name=name, parameters={}, metrics={"metric": 0.5})
        for name in BASELINE_MODEL_NAMES
    )


def _candidate_evidence_files(tmp_path: Path) -> tuple[Path, ...]:
    oof_path = tmp_path / "oof_predictions.csv"
    oof_path.write_text(
        "account_id,variant_id,feature_view,configuration_id,repeat_index,fold_index,probability\n",
        encoding="utf-8",
        newline="\n",
    )
    diagnostics_path = tmp_path / "fold_diagnostics.json"
    diagnostics_path.write_text(
        json.dumps({"fit_count": 210, "fits": [{} for _ in range(210)]}),
        encoding="utf-8",
    )
    summary = {
        "data": {
            "development_rows": 24_000,
            "holdout_evaluated": False,
            "n_repeats": 3,
            "n_splits": 5,
            "partition": "development",
        },
        "fit_budget": {"completed_fold_fits": 210, "maximum_fold_fits": 210},
        "runtime_artifacts": {
            "contains_holdout_rows": False,
            "contains_fitted_models": False,
            "oof_prediction_rows": 1_008_000,
            "oof_predictions_sha256": hashlib.sha256(oof_path.read_bytes()).hexdigest(),
            "fold_diagnostics_sha256": hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
        },
    }
    summary_path = tmp_path / "summary.json"
    summary_bytes = json.dumps(summary, sort_keys=True).encode("utf-8")
    summary_path.write_bytes(summary_bytes)
    report_path = tmp_path / "candidate-report.md"
    report_path.write_text(
        "# Governed CatBoost candidate report\n\n"
        f"Deterministic summary SHA-256:** `{hashlib.sha256(summary_bytes).hexdigest()}`\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary_path, report_path, oof_path, diagnostics_path


def _evidence_files(tmp_path: Path) -> tuple[Path, ...]:
    summary_path = tmp_path / "summary.json"
    report_path = tmp_path / "baseline-report.md"
    oof_path = tmp_path / "oof_predictions.csv"
    diagnostics_path = tmp_path / "logistic_fold_diagnostics.json"
    rows = ["account_id,model_id,repeat_index,fold_index,prediction_kind,score"]
    for model_id, prediction_kind in (
        ("fold_prevalence", "probability"),
        ("repayment_burden_rule", "risk_score"),
        ("logistic_l2", "probability"),
    ):
        for repeat_index in range(3):
            for account_id in range(1, 6):
                rows.append(
                    f"{account_id},{model_id},{repeat_index},{account_id - 1},{prediction_kind},0.5"
                )
    oof_bytes = ("\n".join(rows) + "\n").encode("utf-8")
    oof_path.write_bytes(oof_bytes)
    diagnostics = {
        "schema_version": "1.0.0",
        "model_id": "logistic_l2",
        "transformed_feature_names": ["feature"],
        "folds": [
            {
                "repeat_index": repeat_index,
                "fold_index": fold_index,
                "iterations": 1,
                "intercept": 0.0,
                "coefficients": [0.1],
            }
            for repeat_index in range(3)
            for fold_index in range(5)
        ],
    }
    diagnostics_bytes = (json.dumps(diagnostics, sort_keys=True) + "\n").encode("utf-8")
    diagnostics_path.write_bytes(diagnostics_bytes)
    summary = {
        "data": {
            "development_rows": 5,
            "holdout_evaluated": False,
            "n_repeats": 3,
            "n_splits": 5,
        },
        "models": {model_id: {} for model_id in BASELINE_MODEL_NAMES},
        "runtime_artifacts": {
            "contains_fitted_models": False,
            "contains_holdout_rows": False,
            "oof_predictions_sha256": hashlib.sha256(oof_bytes).hexdigest(),
            "logistic_diagnostics_sha256": hashlib.sha256(diagnostics_bytes).hexdigest(),
        },
    }
    summary_bytes = (json.dumps(summary, sort_keys=True) + "\n").encode("utf-8")
    summary_path.write_bytes(summary_bytes)
    report_path.write_text(
        "# Governed baseline experiment report\n\n"
        "**Deterministic summary SHA-256:** "
        f"`{hashlib.sha256(summary_bytes).hexdigest()}`\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary_path, report_path, oof_path, diagnostics_path


def _rebind_evidence(
    summary_path: Path,
    report_path: Path,
    oof_path: Path,
    diagnostics_path: Path,
) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["runtime_artifacts"]["oof_predictions_sha256"] = hashlib.sha256(
        oof_path.read_bytes()
    ).hexdigest()
    summary["runtime_artifacts"]["logistic_diagnostics_sha256"] = hashlib.sha256(
        diagnostics_path.read_bytes()
    ).hexdigest()
    summary_bytes = (json.dumps(summary, sort_keys=True) + "\n").encode("utf-8")
    summary_path.write_bytes(summary_bytes)
    report_path.write_text(
        "# Governed baseline experiment report\n\n"
        "**Deterministic summary SHA-256:** "
        f"`{hashlib.sha256(summary_bytes).hexdigest()}`\n",
        encoding="utf-8",
        newline="\n",
    )
