"""Independent reference calculations and contract tests, without model access."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from credit_risk.portfolio.planning import (
    MAX_PER_ARM,
    PlanningError,
    main,
    minimum_detectable_effect,
    sample_size,
)


def test_published_r_reference_sample_size():
    # R stats documentation: p1=.50, p2=.75, power=.90 -> n=76.7 per group.
    # Complement both outcomes to express the same calculation as a reduction.
    result = sample_size(0.50, 0.25, power=0.90)
    assert result.continuous_required_per_arm == pytest.approx(76.7, abs=0.01)
    assert result.analysable_per_arm == result.recruited_per_arm == 77
    assert result.total_recruitment == 154


def test_published_r_reference_mde():
    # R documentation: n=50, p1=.5, power=.90 -> p2=.8026.
    result = minimum_detectable_effect(0.5, 50, power=0.90)
    assert result.treatment_rate == pytest.approx(1 - 0.8026, abs=0.00005)
    assert result.absolute_reduction == pytest.approx(0.3026, abs=0.00005)


def test_published_r_small_effect_reference():
    # R documentation: p1=.5, p2=.501, alpha=.001, power=.90 -> n=10451937.
    result = sample_size(0.5, 0.001, alpha=0.001, power=0.90)
    assert result.continuous_required_per_arm == pytest.approx(10451937, abs=1)


def test_hypothetical_example_rounding_and_round_trip():
    result = sample_size(0.30, 0.03, attrition=0.10)
    assert (result.analysable_per_arm, result.recruited_per_arm, result.total_recruitment) == (
        3554,
        3949,
        7898,
    )
    assert result.reduction_percentage_points == 3
    assert result.relative_reduction == pytest.approx(0.10)
    reverse = minimum_detectable_effect(0.30, result.recruited_per_arm, attrition=0.10)
    assert reverse.analysable_per_arm == math.floor(3949 * 0.9)
    assert reverse.absolute_reduction <= 0.03
    assert reverse.absolute_reduction == pytest.approx(0.03, abs=0.00001)
    assert reverse.warnings == result.warnings


def test_expected_sensitivity():
    ordinary = sample_size(0.30, 0.03)
    assert sample_size(0.30, 0.015).recruited_per_arm > ordinary.recruited_per_arm
    assert sample_size(0.30, 0.03, power=0.90).recruited_per_arm > ordinary.recruited_per_arm
    assert sample_size(0.30, 0.03, alpha=0.01).recruited_per_arm > ordinary.recruited_per_arm
    assert sample_size(0.30, 0.03, attrition=0.20).recruited_per_arm > ordinary.recruited_per_arm
    ordinary_mde = minimum_detectable_effect(0.30, 4000).absolute_reduction
    assert minimum_detectable_effect(0.30, 8000).absolute_reduction < ordinary_mde
    assert minimum_detectable_effect(0.30, 4000, power=0.90).absolute_reduction > ordinary_mde
    assert minimum_detectable_effect(0.30, 4000, attrition=0.2).absolute_reduction > ordinary_mde


@pytest.mark.parametrize(
    "options",
    [
        {"baseline_rate": 0},
        {"baseline_rate": 1},
        {"baseline_rate": float("nan")},
        {"baseline_rate": True},
        {"baseline_rate": "0.3"},
        {"absolute_reduction": 0},
        {"absolute_reduction": -0.01},
        {"absolute_reduction": 0.4},
        {"absolute_reduction": float("inf")},
        {"absolute_reduction": 1e-300},
        {"alpha": 0},
        {"alpha": 1},
        {"alpha": 1e-300},
        {"alpha": float("-inf")},
        {"power": 0.5},
        {"power": 1},
        {"power": float("nan")},
        {"attrition": -1},
        {"attrition": 1},
        {"attrition": float("nan")},
        {"attrition": 1 - 1e-16},
    ],
)
def test_invalid_size_requests(options):
    arguments = {"baseline_rate": 0.3, "absolute_reduction": 0.03, **options}
    with pytest.raises(PlanningError):
        sample_size(**arguments)


@pytest.mark.parametrize("count", [True, 0, 1, -4, 2.5, float("inf"), MAX_PER_ARM + 1])
def test_invalid_mde_counts(count):
    with pytest.raises(PlanningError, match="n_per_arm"):
        minimum_detectable_effect(0.3, count)


def test_unattainable_mde_and_attrition():
    with pytest.raises(PlanningError, match="unattainable"):
        minimum_detectable_effect(0.1, 2)
    with pytest.raises(PlanningError, match="fewer than two"):
        minimum_detectable_effect(0.3, 3, attrition=0.9)


def test_boundary_zero_treatment_and_weak_approximation():
    result = sample_size(0.1, 0.1)
    assert result.treatment_rate == 0
    assert any("below 10" in warning for warning in result.warnings)
    assert sample_size(0.9, 0.9, alpha=0.9, power=0.51).analysable_per_arm == 2


def test_cli_output_and_error_contract(capsys):
    common = ["--baseline-rate", ".3", "--attrition", ".1"]
    assert main(["sample-size", *common, "--absolute-reduction", ".03", "--json"]) == 0
    value = json.loads(capsys.readouterr().out)
    assert value["label"] == "planning_estimate_not_observed_effect"
    assert value["recruited_per_arm"] == 3949
    assert main(["mde", *common, "--n-per-arm", "3949"]) == 0
    assert "PLANNING ESTIMATE" in capsys.readouterr().out
    with pytest.raises(SystemExit) as caught:
        main(["sample-size", "--baseline-rate", "nan", "--absolute-reduction", ".03"])
    assert caught.value.code == 2
    assert "finite number" in capsys.readouterr().err


def test_direct_execution_without_site_packages_data_or_network(tmp_path):
    script = Path(__file__).resolve().parents[3] / "src/credit_risk/portfolio/planning.py"
    # -I ignores PYTHONPATH; -S excludes site packages. The audit guard prohibits
    # socket operations and repository reads outside this standalone source.
    guard = """
import runpy, sys
from pathlib import Path
script = Path(sys.argv[1]).resolve()
repo = script.parents[3]
stdlib = Path(sys.base_prefix).resolve()
def audit(event, args):
    if event.startswith('socket.'):
        raise AssertionError('network access')
    if event == 'open' and isinstance(args[0], str):
        path = Path(args[0]).resolve()
        if path.is_relative_to(repo) and path != script and not path.is_relative_to(stdlib):
            raise AssertionError('repository data or model access')
sys.addaudithook(audit)
sys.argv = [str(script), 'sample-size', '--baseline-rate', '.3',
            '--absolute-reduction', '.03', '--json']
runpy.run_path(str(script), run_name='__main__')
"""
    result = subprocess.run(
        [sys.executable, "-I", "-S", "-c", guard, str(script)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(result.stdout)["analysable_per_arm"] == 3554
