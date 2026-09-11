"""Permanent tombstone for the consumed one-time final-test evaluation."""

from __future__ import annotations

from typing import Literal, Never

from pydantic import BaseModel, ConfigDict, Field

CONSUMED_EVALUATION_MESSAGE = (
    "Final-test evaluation was permanently consumed; see "
    "reports/modeling/final_test_v1/evaluation-completed.json. "
    "No reevaluation is permitted."
)


class FinalTestWorkflowError(RuntimeError):
    """Raised when a caller attempts to rerun the consumed final test."""


class FinalTestApproval(BaseModel):
    """Immutable approval schema retained for historical evidence validation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, protected_namespaces=())

    schema_version: Literal["1.0.0"]
    approval_id: Literal["final_test_v1_approval"]
    status: Literal["approved_once"]
    frozen_authorization_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    workflow_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    maximum_evaluations: Literal[1]
    training: Literal["prohibited"]
    refitting: Literal["prohibited"]
    retuning: Literal["prohibited"]
    force_override: Literal["prohibited"]
    dirty_execution: Literal["prohibited"]


def run_final_test(*_args: object, **_kwargs: object) -> Never:
    """Reject every replay attempt before touching data, models, or caller paths."""

    raise FinalTestWorkflowError(CONSUMED_EVALUATION_MESSAGE)
