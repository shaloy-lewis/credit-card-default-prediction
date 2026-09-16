"""Allowlisted JSON logging for inference operations."""

from __future__ import annotations

import json
import logging
import sys

LOGGER = logging.getLogger("credit_risk.inference")
if not LOGGER.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    LOGGER.addHandler(handler)
LOGGER.setLevel(logging.INFO)
LOGGER.propagate = False

ALLOWED_LOG_FIELDS = {
    "event",
    "route",
    "operation",
    "status",
    "trace_id",
    "batch_id",
    "model_id",
    "bundle_id",
    "policy_id",
    "row_count",
    "rejection_count",
    "duration_ms",
}


def emit_event(event: str, **fields: object) -> None:
    """Emit one JSON object containing metadata from the reviewed allowlist only."""

    unexpected = set(fields) - (ALLOWED_LOG_FIELDS - {"event"})
    if unexpected:
        raise ValueError(f"Inference log fields are not allowlisted: {sorted(unexpected)}")
    payload = {"event": event, **fields}
    LOGGER.info(json.dumps(payload, sort_keys=True, separators=(",", ":")))
