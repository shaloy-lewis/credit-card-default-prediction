"""Idempotent monthly batch-scoring tests."""

from __future__ import annotations

import csv
import io
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
        bands = tuple(
            "standard" if probability < 0.3 else "elevated" if probability < 0.6 else "high"
            for probability in probabilities
        )
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


@pytest.mark.parametrize(
    ("content", "message"),
    (
        (b"", "empty"),
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
