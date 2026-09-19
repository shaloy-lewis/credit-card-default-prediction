"""Guard the source repository against generated and retired content."""

from __future__ import annotations

import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).parents[2]


def test_tracked_inventory_excludes_generated_binaries_and_retired_packages() -> None:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = set(result.stdout.splitlines())

    forbidden_prefixes = (
        "catboost_info/",
        "experiment/",
        "src/credit_risk/components/",
        "src/credit_risk/pipeline/",
    )
    forbidden_suffixes = (".ipynb", ".pkl", ".cbm")
    forbidden_exact = {"src/credit_risk/artifacts.py"}
    forbidden_exact.update(
        {
            "src/credit_risk/modeling/candidate_contracts.py",
            "src/credit_risk/modeling/candidate_evidence.py",
            "src/credit_risk/modeling/candidate_selection.py",
            "src/credit_risk/modeling/candidate_workflow.py",
            "src/credit_risk/modeling/candidates.py",
            "src/credit_risk/modeling/checkpoints.py",
            "src/credit_risk/modeling/workflow.py",
        }
    )

    violations = sorted(
        path
        for path in tracked
        if path in forbidden_exact
        or path.startswith(forbidden_prefixes)
        or path.endswith(forbidden_suffixes)
    )
    assert violations == []
