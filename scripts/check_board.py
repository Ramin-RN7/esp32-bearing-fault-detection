#!/usr/bin/env python3
"""Check what a board computed for A5 against the frozen reference (docs/DECISION_CONTRACT.md).

    python scripts/check_board.py board/H_13_0.json board/O_8_1.json ...

Each file holds one recording, as the board wrote it (the format is in docs/DECISION_CONTRACT.md, section 6):
the model it used, and for each of the 9 windows its features and P(fault), optionally with the PCM and filter
summaries and the model stages computed from the golden features. The four golden recordings are checked stage
by stage; the others by features, probabilities and the decision (reference/a5/window_features.csv and
decisions.csv). Every check prints as a fraction of its tolerance: 1.0 is at the limit, above it fails.
Exits with status 1 if anything fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bearing_audio import reference as R           # noqa: E402
from bearing_audio.features import FEATURE_NAMES   # noqa: E402

REF = ROOT / "reference" / "a5"


def expected_windows(name: str, model: str, golden: dict | None, table: dict) -> list[dict]:
    """The reference record of each window: golden stages when available, else features and P(fault)."""
    lobo = f"models/{table['decisions'][name]['fold_model']}.json"
    if model not in (lobo, "models/all_bearings.json"):
        sys.exit(f"{name}: model {model} is neither its fold model ({lobo}) nor models/all_bearings.json")
    out = []
    for i in range(9):
        if golden:
            w = dict(golden["windows"][i])
            if model == "models/all_bearings.json":
                w["probability"] = golden["all_bearings"]["probabilities"][i]
        else:
            row = table["windows"][(name, i + 1)]
            w = {"features": [float(row[n]) for n in FEATURE_NAMES], "lag_gap": float(row["lag_gap"]),
                 "probability": float(row["p_lobo" if model == lobo else "p_all_bearings"])}
        out.append(w)
    return out


def check_file(path: Path, spread, table: dict) -> tuple[bool, list[str]]:
    board = json.loads(path.read_text())
    name, model = board["recording"], board["model"]
    gpath = REF / "golden" / f"{name}.json"
    golden = json.loads(gpath.read_text()) if gpath.exists() else None
    expected = expected_windows(name, model, golden, table)
    if len(board["windows"]) != 9:
        return False, [f"{name}: expected 9 windows, got {len(board['windows'])}"]
    results: dict[str, float] = {}

    def note(check, value):
        results[check] = max(results.get(check, 0.0), value)

    for bw, ew in zip(board["windows"], expected):
        stages = {k: v for k, v in bw.items()
                  if k in ("features", "probability") or (golden and k in ("pcm", "highpass", "preemphasis", "notch"))}
        for c, v in R.check_window(stages, ew, spread).items():
            note(f"window {bw['window']} {c}", v)
    for mw in board.get("model_from_golden", []):
        if not golden or model != golden["model"]:
            break
        for c, v in R.check_model(mw, golden["windows"][mw["window"] - 1]).items():
            note(f"window {mw['window']} model {c}", v)
    dec = table["decisions"][name]
    ref_score = float(dec["lobo_score" if model != "models/all_bearings.json" else "all_bearings_score"])
    ref_decision = int(dec["lobo_decision" if model != "models/all_bearings.json" else "all_bearings_decision"])
    note("recording score", abs(board["score"] - ref_score) / R.PROBABILITY_TOLERANCE["recording"])
    note("recording decision", 0.0 if int(board["decision"]) == ref_decision else float("inf"))

    failed = sorted(((v, c) for c, v in results.items() if v > 1.0), reverse=True)
    worst_value, worst = max((v, c) for c, v in results.items())
    lines = [f"{name} ({model}): {'FAIL' if failed else 'PASS'}; {len(results)} checks, worst {worst} "
             f"at {worst_value:.3g} of its tolerance; decision {board['decision']} (reference {ref_decision})"]
    lines += [f"    {c}: {v:.3g}" for v, c in failed[:20]]
    if len(failed) > 20:
        lines.append(f"    ... and {len(failed) - 20} more")
    return not failed, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()
    spread = np.asarray(json.loads((REF / "contract.json").read_text())["features"]["spread"])
    with open(REF / "decisions.csv", newline="") as f:
        decisions = {row["recording"]: row for row in csv.DictReader(f)}
    with open(REF / "window_features.csv", newline="") as f:
        windows = {(row["recording"], int(row["window"])): row for row in csv.DictReader(f)}
    table = {"decisions": decisions, "windows": windows}
    ok = True
    for path in a.files:
        passed, lines = check_file(path, spread, table)
        ok &= passed
        print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
