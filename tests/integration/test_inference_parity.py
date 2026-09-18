"""Real-bundle parity across shared, batch, HTTP, and demo-client adapters."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from catboost import CatBoostClassifier
from fastapi.testclient import TestClient

import credit_risk.inference.client as client
from api import create_app
from credit_risk.inference.batch import parse_batch_csv, run_batch
from credit_risk.inference.contracts import load_inference_config
from credit_risk.inference.engine import InferenceEngine

pytestmark = [pytest.mark.integration, pytest.mark.artifact]
FIXTURE = Path("tests/fixtures/inference_batch_v1.csv")


class _ClientResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def __enter__(self) -> _ClientResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def test_real_bundle_batch_api_and_demo_client_are_prediction_equivalent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        CatBoostClassifier,
        "fit",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("fit is prohibited")),
    )
    config = load_inference_config()
    parsed = parse_batch_csv(FIXTURE.read_bytes(), config)
    engine = InferenceEngine()
    offline = engine.score(parsed.features)
    batch = run_batch(
        input_path=FIXTURE,
        as_of_date="2026-09-30",
        snapshot_id="synthetic-parity-v1",
        output_root=tmp_path / "batches",
        config=config,
        engine=engine,
    )
    batch_rows = {
        row["account_id"]: row
        for row in csv.DictReader((batch.run_root / "scores.csv").open(encoding="utf-8"))
    }

    assert batch.status == "completed"
    assert batch.valid_rows == 20
    assert sum(row["selected_for_review"] == "true" for row in batch_rows.values()) == 2
    with TestClient(create_app()) as http:
        for position, account_id in enumerate(parsed.account_ids):
            payload = {
                name: int(parsed.features.loc[account_id, name])
                for name in config.prediction.feature_order
            }
            response = http.post(
                "/v1/predict",
                json=payload,
                headers={"X-Request-ID": f"parity-{position}"},
            )
            assert response.status_code == 200
            api_payload = response.json()
            batch_row = batch_rows[account_id]
            assert float(batch_row["probability_of_default"]) == pytest.approx(
                offline.probabilities[position], abs=1e-15
            )
            assert api_payload["probability_of_default"] == pytest.approx(
                offline.probabilities[position], abs=config.prediction.api_batch_absolute_tolerance
            )
            assert api_payload["risk_band"] == batch_row["risk_band"]
            assert [item["category"] for item in api_payload["reasons"]] == [
                batch_row["primary_reason_category"],
                batch_row["secondary_reason_category"],
            ]
            assert [item["direction"] for item in api_payload["reasons"]] == [
                batch_row["primary_reason_direction"],
                batch_row["secondary_reason_direction"],
            ]

        first_payload = {
            name: int(parsed.features.iloc[0][name]) for name in config.prediction.feature_order
        }
        expected = http.post(
            "/v1/predict", json=first_payload, headers={"X-Request-ID": "demo-client"}
        ).json()

    monkeypatch.setattr(
        client.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: _ClientResponse(expected),
    )
    assert client.predict_v1(first_payload, request_id="demo-client") == expected

    first_mtimes = {path.name: path.stat().st_mtime_ns for path in batch.run_root.iterdir()}
    reused = run_batch(
        input_path=FIXTURE,
        as_of_date="2026-09-30",
        snapshot_id="synthetic-parity-v1",
        output_root=tmp_path / "batches",
        config=config,
        engine=engine,
    )
    assert reused.reused is True
    assert first_mtimes == {path.name: path.stat().st_mtime_ns for path in batch.run_root.iterdir()}
