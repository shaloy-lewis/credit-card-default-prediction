"""Regression tests for the permanently consumed final-test tombstone."""

from __future__ import annotations

from pathlib import Path

import pytest

from credit_risk.modeling.final_test_workflow import (
    CONSUMED_EVALUATION_MESSAGE,
    FinalTestWorkflowError,
    run_final_test,
)


class _ExplodingPath:
    """Fail if the tombstone tries to inspect a caller-controlled path."""

    def __fspath__(self) -> str:
        raise AssertionError("the retired evaluator accessed a path")

    def __str__(self) -> str:
        raise AssertionError("the retired evaluator formatted a path")


def test_final_test_is_permanently_consumed() -> None:
    with pytest.raises(FinalTestWorkflowError, match="permanently consumed") as error:
        run_final_test()

    assert str(error.value) == CONSUMED_EVALUATION_MESSAGE
    assert "evaluation-completed.json" in str(error.value)
    assert "No reevaluation is permitted" in str(error.value)


def test_arbitrary_fresh_paths_cannot_bypass_consumption() -> None:
    inaccessible = _ExplodingPath()

    with pytest.raises(FinalTestWorkflowError, match="permanently consumed"):
        run_final_test(
            data_root=inaccessible,
            authorization_path=inaccessible,
            approval_path=inaccessible,
            bundle_root=inaccessible,
            runtime_root=Path("a-completely-fresh-runtime"),
            output_root=Path("a-completely-fresh-output"),
        )


def test_positional_arguments_are_rejected_before_access() -> None:
    with pytest.raises(FinalTestWorkflowError, match="No reevaluation is permitted"):
        run_final_test(_ExplodingPath(), object())
