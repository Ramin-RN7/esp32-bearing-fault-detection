"""The generated report describes what was measured, and the published report matches its data."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("evaluate_protocols", ROOT / "scripts" / "evaluate_protocols.py")
report = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(report)


def test_small_protocol_differences_are_called_small():
    diffs = {"a8_gmm": -0.0004, "a16_mlp": 0.0104, "baseline_rms_kurtosis": -0.0183}     # published 200-split run
    assert "every mean difference is below 0.02" in report.protocol_difference_note(diffs, 200)


def test_large_protocol_difference_is_not_called_small():
    diffs = {"a8_gmm": 0.25, "a16_mlp": 0.0833, "baseline_rms_kurtosis": 0.0}           # a one-split run
    text = report.protocol_difference_note(diffs, 1)
    assert "below 0.02" not in text
    assert "0.250" in text


def test_difference_that_prints_as_0_020_is_not_called_below_0_02():
    text = report.protocol_difference_note({"a8_gmm": 0.0199, "a16_mlp": 0.0, "baseline_rms_kurtosis": 0.0}, 20)
    assert "below 0.02" not in text


def test_published_report_matches_its_json():
    res = json.loads((ROOT / "results" / "protocol_comparison.json").read_text())
    assert report.markdown(res) == (ROOT / "results" / "protocol_comparison.md").read_text()
