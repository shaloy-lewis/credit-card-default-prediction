"""Integrity proof that the one-time test gates were frozen without execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AUTHORIZATION_PATH = REPOSITORY_ROOT / "configs" / "modeling" / "final_test_v1.json"

# Change only after a separately reviewed selection release and explicit gate review.
EXPECTED_SHA256 = "58f9566e37883d8955d4b10b3b82de1fa164909adae79985bf3eb668ce1c9251"


def test_final_test_authorization_is_frozen_validation_evidence_only() -> None:
    content = AUTHORIZATION_PATH.read_bytes()
    assert hashlib.sha256(content).hexdigest() == EXPECTED_SHA256
    authorization = json.loads(content)

    assert authorization["status"] == "frozen_not_executed"
    assert authorization["selection_evidence"] == {
        "evidence_commit": "d334b886ae5ac34f1394f17683eaa2eaddfea1ea",
        "implementation_commit": "f7c99f257fe756f6db6bac449a7ef4f48a899ea4",
        "manifest_sha256": ("df5ce6ce07b268f57fa3bf72c97cd32f8ebb66695d7157139942c91e46d7cd88"),
        "model_sha256": "844ec1c33a894cbf01dcaf8672443fa38d86a06b8965ed729afccaf08f24d88c",
        "report_sha256": "16c8748e76002ebedd5c41938df7364e493af590461d03a926a2aab3d801cee1",
        "selected_model_id": "catboost_fixed",
        "summary_sha256": "8c11b1d443c782a8ef14aa3e708e3fffa064ecb4c9fe58d3e51a6effa46efbd7",
    }
    assert authorization["frozen_gates"] == pytest.approx(
        {
            "minimum_average_precision": 0.5265104548302114,
            "maximum_brier_score": 0.15353854208377515,
            "minimum_lift_at_0_1": 2.910922787193974,
        }
    )
    assert authorization["test_contract"]["required_unique_accounts"] == 6000
    assert authorization["test_contract"]["maximum_evaluations"] == 1
    assert authorization["test_contract"]["training"] == "prohibited"
    assert authorization["test_contract"]["refitting"] == "prohibited"
    assert authorization["test_contract"]["retuning"] == "prohibited"
    assert authorization["test_contract"]["force_override"] == "prohibited"
    assert authorization["execution"] == {
        "authorized": False,
        "holdout_loaded_during_freeze": False,
        "requires_separate_explicit_request": True,
    }
