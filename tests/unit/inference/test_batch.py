"""Idempotent monthly batch-scoring tests."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pytest

from credit_risk.inference.batch import (
    BatchInferenceError,
    parse_batch_csv,
    run_batch,
    verify_batch_run,
)
from credit_risk.inference.contracts import InferenceConfig, load_inference_config
from credit_risk.inference.engine import InferenceResult, ReasonAttribution
from credit_risk.modeling.contracts import PREDICTOR_COLUMNS
from credit_risk.modeling.risk_policy import risk_band


class _Engine:
    def score(self, frame: object) -> InferenceResult:
        rows = len(frame)  # type: ignore[arg-type]
        probabilities = np.linspace(0.1, 0.9, rows)
        reasons = tuple(
            (
                ReasonAttribution("repayment_status", "risk_increasing", 0.5),
                ReasonAttribution("credit_capacity", "risk_mitigating", -0.25),
            )
            for _ in range(rows)
        )
        thresholds = load_inference_config().prediction.risk_band_thresholds
        bands = tuple(risk_band(float(probability), thresholds) for probability in probabilities)
        return InferenceResult(
            probabilities=probabilities,
            risk_bands=bands,  # type: ignore[arg-type]
            reasons=reasons,
            category_contributions=np.zeros((rows, 4)),
            max_additivity_error=0.0,
            max_probability_error=0.0,
        )


@pytest.fixture
def config() -> InferenceConfig:
    return load_inference_config()


def _values(account_id: str, *, credit_limit: str = "100000") -> list[str]:
    payload = {
        "credit_limit_ntd": credit_limit,
        **{f"repayment_status_lag_{lag}": "0" for lag in range(6)},
        **{f"bill_amount_ntd_lag_{lag}": "4000" for lag in range(6)},
        **{f"payment_amount_ntd_lag_{lag}": "1500" for lag in range(6)},
    }
    return [account_id, *(payload[column] for column in PREDICTOR_COLUMNS)]


def _csv(config: InferenceConfig, rows: list[list[str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(config.input.columns)
    writer.writerows(rows)
    return stream.getvalue().encode()


def test_parse_rejects_duplicate_and_invalid_rows(config: InferenceConfig) -> None:
    parsed = parse_batch_csv(
        _csv(
            config,
            [
                _values("duplicate"),
                _values("duplicate"),
                _values("valid"),
                _values("unsafe id"),
                _values("bad-limit", credit_limit="0"),
            ],
        ),
        config,
    )

    assert parsed.input_rows == 5
    assert parsed.account_ids == ("valid",)
    assert [rejection.rule_ids for rejection in parsed.rejections] == [
        ("duplicate_account_id",),
        ("duplicate_account_id",),
        ("invalid_account_id",),
        ("invalid_credit_limit_ntd",),
    ]
    assert parsed.rejections[2].account_id == ""


def test_parse_rejects_wrong_width_rows_without_blocking_valid_rows(
    config: InferenceConfig,
) -> None:
    parsed = parse_batch_csv(
        _csv(
            config,
            [
                _values("valid"),
                _values("missing")[:-1],
                [*_values("extra"), "unexpected"],
                [],
                _values("unsafe id")[:-1],
            ],
        ),
        config,
    )

    assert parsed.input_rows == 5
    assert parsed.account_ids == ("valid",)
    assert [(item.account_id, item.rule_ids) for item in parsed.rejections] == [
        ("missing", ("invalid_column_count",)),
        ("extra", ("invalid_column_count",)),
        ("", ("invalid_account_id", "invalid_column_count")),
        ("", ("invalid_account_id", "invalid_column_count")),
    ]


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (b"", "empty"),
        (_csv(load_inference_config(), []), "no account rows"),
        (b"wrong,header\n1,2\n", "headers"),
        (b"\xff", "UTF-8"),
        (b'account_id,"unterminated\n', "malformed CSV"),
    ),
)
def test_parse_rejects_file_level_errors(
    config: InferenceConfig, content: bytes, message: str
) -> None:
    with pytest.raises(BatchInferenceError, match=message):
        parse_batch_csv(content, config)


def test_run_batch_ranks_selects_and_reuses_without_rewrite(
    tmp_path: Path, config: InferenceConfig
) -> None:
    source = tmp_path / "input.csv"
    source.write_bytes(_csv(config, [_values(f"acct-{index:02d}") for index in range(20)]))
    output_root = tmp_path / "output"

    first = run_batch(
        input_path=source,
        as_of_date="2026-09-30",
        snapshot_id="monthly-v1",
        output_root=output_root,
        config=config,
        engine=_Engine(),  # type: ignore[arg-type]
    )
    mtimes = {path.name: path.stat().st_mtime_ns for path in first.run_root.iterdir()}
    second = run_batch(
        input_path=source,
        as_of_date="2026-09-30",
        snapshot_id="monthly-v1",
        output_root=output_root,
        config=config,
        engine=_Engine(),  # type: ignore[arg-type]
    )

    assert first.status == "completed"
    assert first.exit_code == 0
    assert first.valid_rows == 20
    assert second.reused is True
    assert mtimes == {path.name: path.stat().st_mtime_ns for path in first.run_root.iterdir()}
    manifest = verify_batch_run(first.run_root, config=config, expected_batch_id=first.batch_id)
    assert manifest["policy"]["selected_rows"] == 2
    score_rows = list(csv.DictReader((first.run_root / "scores.csv").open(encoding="utf-8")))
    assert [row["portfolio_rank"] for row in score_rows] == [str(index) for index in range(1, 21)]
    assert sum(row["selected_for_review"] == "true" for row in score_rows) == 2
    assert score_rows[0]["account_id"] == "acct-19"
    assert "acct-" not in (first.run_root / "manifest.json").read_text(encoding="utf-8")


def test_partial_and_all_invalid_batches_publish_reviewable_evidence(
    tmp_path: Path, config: InferenceConfig
) -> None:
    partial_source = tmp_path / "partial.csv"
    partial_source.write_bytes(
        _csv(config, [_values("valid"), _values("invalid", credit_limit="0")])
    )
    partial = run_batch(
        input_path=partial_source,
        as_of_date="2026-09-30",
        snapshot_id="partial",
        output_root=tmp_path / "output",
        config=config,
        engine=_Engine(),  # type: ignore[arg-type]
    )
    assert (partial.status, partial.exit_code, partial.valid_rows, partial.rejected_rows) == (
        "completed_with_rejections",
        3,
        1,
        1,
    )

    invalid_source = tmp_path / "invalid.csv"
    invalid_source.write_bytes(_csv(config, [_values("invalid", credit_limit="0")]))
    invalid = run_batch(
        input_path=invalid_source,
        as_of_date="2026-09-30",
        snapshot_id="invalid",
        output_root=tmp_path / "output",
        config=config,
        engine=_Engine(),  # type: ignore[arg-type]
    )
    assert (invalid.status, invalid.exit_code, invalid.valid_rows, invalid.rejected_rows) == (
        "failed",
        1,
        0,
        1,
    )
    assert len((invalid.run_root / "scores.csv").read_text(encoding="utf-8").splitlines()) == 1


def test_changed_or_corrupt_existing_run_is_not_overwritten(
    tmp_path: Path, config: InferenceConfig
) -> None:
    source = tmp_path / "input.csv"
    source.write_bytes(_csv(config, [_values("acct-1")]))
    result = run_batch(
        input_path=source,
        as_of_date="2026-09-30",
        snapshot_id="conflict",
        output_root=tmp_path / "output",
        config=config,
        engine=_Engine(),  # type: ignore[arg-type]
    )
    (result.run_root / "scores.csv").write_text("corrupt", encoding="utf-8")
    with pytest.raises(BatchInferenceError, match="digest mismatch"):
        run_batch(
            input_path=source,
            as_of_date="2026-09-30",
            snapshot_id="conflict",
            output_root=tmp_path / "output",
            config=config,
            engine=_Engine(),  # type: ignore[arg-type]
        )

    different = tmp_path / "different.csv"
    different.write_bytes(_csv(config, [_values("acct-2")]))
    with pytest.raises(BatchInferenceError, match="batch identity"):
        run_batch(
            input_path=different,
            as_of_date="2026-09-30",
            snapshot_id="conflict",
            output_root=tmp_path / "output",
            config=config,
            engine=_Engine(),  # type: ignore[arg-type]
        )


def test_batch_rejects_unsafe_identifiers_dates_and_missing_files(
    tmp_path: Path, config: InferenceConfig
) -> None:
    source = tmp_path / "input.csv"
    source.write_bytes(_csv(config, [_values("acct-1")]))
    common = {
        "input_path": source,
        "output_root": tmp_path / "output",
        "config": config,
        "engine": _Engine(),
    }
    with pytest.raises(BatchInferenceError, match="Snapshot ID"):
        run_batch(as_of_date="2026-09-30", snapshot_id="unsafe id", **common)  # type: ignore[arg-type]
    for reserved in (".", ".."):
        with pytest.raises(BatchInferenceError, match="Snapshot ID"):
            run_batch(
                input_path=tmp_path / "missing.csv",
                as_of_date="2026-09-30",
                snapshot_id=reserved,
                output_root=tmp_path / "must-not-exist",
                config=config,
                engine=_Engine(),  # type: ignore[arg-type]
            )
    assert not (tmp_path / "must-not-exist").exists()
    with pytest.raises(BatchInferenceError, match="ISO date"):
        run_batch(as_of_date="09/30/2026", snapshot_id="safe", **common)  # type: ignore[arg-type]
    with pytest.raises(BatchInferenceError, match="Unable to read"):
        run_batch(
            input_path=tmp_path / "missing.csv",
            as_of_date="2026-09-30",
            snapshot_id="safe",
            output_root=tmp_path / "output",
            config=config,
            engine=_Engine(),  # type: ignore[arg-type]
        )


def _completed_run(tmp_path: Path, config: InferenceConfig):  # type: ignore[no-untyped-def]
    source = tmp_path / "input.csv"
    source.write_bytes(_csv(config, [_values(f"acct-{index:02d}") for index in range(20)]))
    return source, run_batch(
        input_path=source,
        as_of_date="2026-09-30",
        snapshot_id="reviewed",
        output_root=tmp_path / "output",
        config=config,
        engine=_Engine(),  # type: ignore[arg-type]
    )


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update(schema_version="2.0.0"),
        lambda value: value.update(protocol_id="tampered"),
        lambda value: value.update(status="failed"),
        lambda value: value.update(as_of_date="2026-9-30"),
        lambda value: value.update(snapshot_id=".."),
        lambda value: value.update(config_sha256="0" * 64),
        lambda value: value["model"].update(model_id="different"),
        lambda value: value["policy"].update(selected_rows=999),
        lambda value: value["counts"]["risk_bands"].update(standard=999),
        lambda value: value["counts"]["risk_bands"].pop("standard"),
        lambda value: value.update(input_sha256="0" * 64),
        lambda value: value["privacy"].update(row_level_values_in_manifest=True),
        lambda value: value["explanation"].update(maximum_additivity_error=1.0),
        lambda value: value["outputs"].update(unapproved="0" * 64),
    ),
)
def test_verifier_rejects_manifest_semantic_tampering(
    tmp_path: Path,
    config: InferenceConfig,
    mutation: object,
) -> None:
    source, result = _completed_run(tmp_path, config)
    manifest_path = result.run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutation(manifest)  # type: ignore[operator]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(BatchInferenceError):
        verify_batch_run(result.run_root, config=config, expected_batch_id=result.batch_id)
    with pytest.raises(BatchInferenceError):
        run_batch(
            input_path=source,
            as_of_date="2026-09-30",
            snapshot_id="reviewed",
            output_root=tmp_path / "output",
            config=config,
            engine=_Engine(),  # type: ignore[arg-type]
        )


def test_verifier_rejects_semantically_changed_scores_even_with_updated_digest(
    tmp_path: Path, config: InferenceConfig
) -> None:
    _, result = _completed_run(tmp_path, config)
    scores_path = result.run_root / "scores.csv"
    rows = list(csv.reader(scores_path.open(encoding="utf-8", newline="")))
    rows[3][1] = "true"
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    changed = stream.getvalue().encode()
    scores_path.write_bytes(changed)
    manifest_path = result.run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"]["scores.csv"] = hashlib.sha256(changed).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(BatchInferenceError, match="selection flags"):
        verify_batch_run(result.run_root, config=config, expected_batch_id=result.batch_id)


@pytest.mark.parametrize("equal_absolute_contributions", (False, True))
def test_verifier_rejects_reordered_reasons_even_with_updated_digest(
    tmp_path: Path,
    config: InferenceConfig,
    equal_absolute_contributions: bool,
) -> None:
    _, result = _completed_run(tmp_path, config)
    scores_path = result.run_root / "scores.csv"
    rows = list(csv.reader(scores_path.open(encoding="utf-8", newline="")))
    header = rows[0]
    row = rows[1]
    primary = [
        header.index("primary_reason_category"),
        header.index("primary_reason_direction"),
        header.index("primary_reason_contribution_raw_log_odds"),
    ]
    secondary = [
        header.index("secondary_reason_category"),
        header.index("secondary_reason_direction"),
        header.index("secondary_reason_contribution_raw_log_odds"),
    ]
    if equal_absolute_contributions:
        row[primary[0]], row[primary[1]], row[primary[2]] = (
            "repayment_status",
            "risk_increasing",
            "1",
        )
        row[secondary[0]], row[secondary[1]], row[secondary[2]] = (
            "billing_balance",
            "risk_mitigating",
            "-1",
        )
    else:
        for primary_index, secondary_index in zip(primary, secondary, strict=True):
            row[primary_index], row[secondary_index] = row[secondary_index], row[primary_index]
    _write_rows_and_update_digest(result.run_root, "scores.csv", rows)

    with pytest.raises(BatchInferenceError, match="contribution ordering"):
        verify_batch_run(result.run_root, config=config, expected_batch_id=result.batch_id)


@pytest.mark.parametrize(
    ("column", "changed_value", "message"),
    (
        ("portfolio_rank", "99", "ranks"),
        ("selected_for_review", "yes", "canonical booleans"),
        ("probability_of_default", "nan", "valid range"),
        ("risk_band", "standard", "risk band"),
        ("primary_reason_category", "unknown", "reason categories"),
        ("trace_id", "0" * 32, "trace ID"),
        ("primary_reason_direction", "neutral", "reason direction"),
        ("model_id", "different", "lineage"),
    ),
)
def test_verifier_rejects_changed_score_semantics_with_updated_digest(
    tmp_path: Path,
    config: InferenceConfig,
    column: str,
    changed_value: str,
    message: str,
) -> None:
    _, result = _completed_run(tmp_path, config)
    scores_path = result.run_root / "scores.csv"
    rows = list(csv.reader(scores_path.open(encoding="utf-8", newline="")))
    rows[1][rows[0].index(column)] = changed_value
    _write_rows_and_update_digest(result.run_root, "scores.csv", rows)

    with pytest.raises(BatchInferenceError, match=message):
        verify_batch_run(result.run_root, config=config, expected_batch_id=result.batch_id)


def test_verifier_rejects_changed_rejection_semantics_with_updated_digest(
    tmp_path: Path, config: InferenceConfig
) -> None:
    source = tmp_path / "partial.csv"
    source.write_bytes(_csv(config, [_values("valid"), _values("invalid", credit_limit="0")]))
    result = run_batch(
        input_path=source,
        as_of_date="2026-09-30",
        snapshot_id="partial",
        output_root=tmp_path / "output",
        config=config,
        engine=_Engine(),  # type: ignore[arg-type]
    )
    rejection_path = result.run_root / "rejections.csv"
    rows = list(csv.reader(rejection_path.open(encoding="utf-8", newline="")))
    rows[1][2] = "unknown_rule"
    _write_rows_and_update_digest(result.run_root, "rejections.csv", rows)

    with pytest.raises(BatchInferenceError, match="rule IDs"):
        verify_batch_run(result.run_root, config=config, expected_batch_id=result.batch_id)

    rows[1][1] = "unsafe id"
    rows[1][2] = "invalid_credit_limit_ntd"
    _write_rows_and_update_digest(result.run_root, "rejections.csv", rows)
    with pytest.raises(BatchInferenceError, match="unsafe account ID"):
        verify_batch_run(result.run_root, config=config, expected_batch_id=result.batch_id)


def test_verifier_rejects_malformed_output_with_updated_digest(
    tmp_path: Path, config: InferenceConfig
) -> None:
    _, result = _completed_run(tmp_path, config)
    _write_rows_and_update_digest(result.run_root, "scores.csv", [["wrong", "header"]])

    with pytest.raises(BatchInferenceError, match="headers"):
        verify_batch_run(result.run_root, config=config, expected_batch_id=result.batch_id)


def test_verifier_rejects_extra_directories_and_wrong_file_types(
    tmp_path: Path, config: InferenceConfig
) -> None:
    _, result = _completed_run(tmp_path, config)
    extra = result.run_root / "unapproved"
    extra.mkdir()
    with pytest.raises(BatchInferenceError, match="allowlist"):
        verify_batch_run(result.run_root, config=config)
    extra.rmdir()

    scores = result.run_root / "scores.csv"
    scores.unlink()
    scores.mkdir()
    with pytest.raises(BatchInferenceError, match="allowlist"):
        verify_batch_run(result.run_root, config=config)


def test_verifier_rejects_missing_run_root(tmp_path: Path, config: InferenceConfig) -> None:
    with pytest.raises(BatchInferenceError, match="does not exist"):
        verify_batch_run(tmp_path / "missing", config=config)


def test_verifier_rejects_symlinked_approved_output(
    tmp_path: Path, config: InferenceConfig
) -> None:
    _, result = _completed_run(tmp_path, config)
    scores = result.run_root / "scores.csv"
    target = tmp_path / "scores-target.csv"
    target.write_bytes(scores.read_bytes())
    scores.unlink()
    try:
        scores.symlink_to(target)
    except OSError:
        pytest.skip("Creating symlinks is not permitted in this Windows environment.")

    with pytest.raises(BatchInferenceError, match="allowlist"):
        verify_batch_run(result.run_root, config=config)


def _write_rows_and_update_digest(run_root: Path, filename: str, rows: list[list[str]]) -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows(rows)
    changed = stream.getvalue().encode()
    (run_root / filename).write_bytes(changed)
    manifest_path = run_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["outputs"][filename] = hashlib.sha256(changed).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
