"""Monitoring must detect injected shifts without overstating small samples."""

import copy
import json

import numpy as np
import pytest

from credit_risk.assurance.evidence import EvidenceError
from credit_risk.monitoring.benchmark import check_acceptance, targets
from credit_risk.monitoring.drift import compare, profile
from credit_risk.monitoring.workflow import summarize_events


@pytest.mark.parametrize("values", [np.zeros(400), np.arange(400), np.tile([-2, -1, 0, 9], 100)])
def test_unchanged_reference_is_clear(values):
    reference = profile(values)
    assert compare(reference, values)["distance"] == 0
    assert compare(reference, values)["status"] == "clear"


def test_overflow_and_constant_reference():
    reference = profile(np.zeros(400))
    assert compare(reference, np.ones(400))["status"] == "investigate"
    assert compare(reference, -np.ones(400))["distance"] == 1
    assert compare(reference, [])["status"] == "insufficient_data"
    assert compare(reference, np.ones(199))["status"] == "insufficient_data"


@pytest.mark.parametrize(
    "fraction,status", [(0, "clear"), (0.05, "clear"), (0.1, "warning"), (0.2, "investigate")]
)
def test_alert_boundaries(fraction, status):
    values = np.zeros(1000)
    values[: int(1000 * fraction)] = 1
    assert compare(profile(np.zeros(1000)), values)["status"] == status


def test_categories_preserve_negative_repayment_codes():
    values = np.tile(np.arange(-2, 10), 40)
    ref = profile(values, categorical=True)
    assert compare(ref, values)["distance"] == 0
    assert compare(ref, np.full(480, 9))["status"] == "investigate"


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [[1, 2]]])
def test_invalid_reference(values):
    with pytest.raises(EvidenceError):
        profile(values)


@pytest.mark.parametrize("values", [[-3], [10], [1.2]])
def test_invalid_categories(values):
    with pytest.raises(EvidenceError):
        profile(values, True)


@pytest.mark.parametrize("change", [{"cuts": [0, 0]}, {"counts": [-1, 1]}, {"rows": 0}])
def test_invalid_histograms(change):
    reference = profile(np.zeros(400))
    reference.update(change)
    with pytest.raises(EvidenceError):
        compare(reference, np.zeros(400))


def test_service_counts_no_double_counting_prediction_events():
    events = [
        {"event": "api_request_completed", "status": "200", "duration_ms": 10},
        {"event": "api_request_completed", "status": "422", "duration_ms": 20},
        {"event": "api_request_completed", "status": "500", "duration_ms": 30},
        {"event": "api_prediction_completed", "status": "completed", "duration_ms": 7},
        {"event": "batch_attempt_completed", "status": "failed"},
        {"event": "batch_attempt_completed", "status": "completed"},
        {"event": "service_health_probe", "status": "unavailable"},
    ]
    result = summarize_events([json.dumps(x) for x in events])
    assert result["request_count"] == 3
    assert result["client_error_count"] == result["server_error_count"] == 1
    assert result["batch_failure_count"] == result["health_failure_count"] == 1
    assert result["p95_latency_ms"] == 29
    assert summarize_events([])["status"] == "insufficient_data"


@pytest.mark.parametrize(
    "event",
    [
        {"event": "x", "account_id": "secret"},
        {"event": "api_request_completed", "duration_ms": -1, "status": "200"},
        {"event": "api_request_completed", "duration_ms": True, "status": "200"},
        {"event": "api_request_completed", "duration_ms": 1, "status": "unknown"},
        {"event": "batch_attempt_completed", "status": "unknown"},
        {"event": "service_health_probe", "status": "unknown"},
    ],
)
def test_service_rejects_bad_or_private_events(event):
    with pytest.raises(EvidenceError):
        summarize_events([json.dumps(event)])


def test_malformed_events():
    for value in ["bad", "[]"]:
        with pytest.raises(EvidenceError):
            summarize_events([value])


def test_rehearsal_limits_and_separate_acceptance():
    trials = [
        {"api_p95_ms": i, "batch_seconds": i * 2, "recovery_seconds": i * 3} for i in [1, 2, 3]
    ]
    limits = targets(trials)
    assert limits == {"api_p95_ms": 6, "batch_seconds": 12, "recovery_seconds": 18}
    check_acceptance(trials[-1], limits)
    over = copy.deepcopy(limits)
    over["api_p95_ms"] += 0.01
    with pytest.raises(EvidenceError):
        check_acceptance(over, limits)
    with pytest.raises(EvidenceError):
        targets(trials[:2])
    with pytest.raises(EvidenceError):
        check_acceptance({}, limits)
