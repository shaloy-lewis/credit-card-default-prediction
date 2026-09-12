"""Controlled Release A repository fixtures."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from credit_risk.release.contracts import load_release_config

SOURCE_ROOT = Path(__file__).resolve().parents[3]
SOURCE_CONFIG = SOURCE_ROOT / "configs" / "releases" / "release_a_v1.json"


def copy_release_repository(root: Path) -> Path:
    """Copy only reviewed aggregate inputs into an isolated repository."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    config = load_release_config(SOURCE_CONFIG)
    config_path = root / "configs" / "releases" / "release_a_v1.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_CONFIG, config_path)
    for reference in config.source_artifacts.values():
        destination = root / reference.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE_ROOT / reference.path, destination)
    uncertainty = root / config.uncertainty_source.path
    uncertainty.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE_ROOT / config.uncertainty_source.path, uncertainty)
    return config_path


def config_payload() -> dict[str, Any]:
    return json.loads(SOURCE_CONFIG.read_bytes())


def write_config(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def fake_data_result(root: Path) -> SimpleNamespace:
    selection = json.loads((root / "reports/modeling/selection_v1/summary.json").read_bytes())
    lineage = selection["reproducibility"]["data_lineage"]
    return SimpleNamespace(
        source_sha256=lineage["source_sha256"],
        dataset_manifest_sha256=lineage["dataset_manifest_sha256"],
        canonical_sha256=lineage["canonical_sha256"],
        quality_report_sha256=lineage["quality_report_sha256"],
        split_config_sha256=lineage["split_config_sha256"],
        assignment_sha256=lineage["assignment_sha256"],
        split_manifest_sha256=lineage["split_manifest_sha256"],
        reviewed_lock_verified=True,
    )
