#!/usr/bin/env python3
"""Check what a board computed for A5 against the frozen reference (docs/DECISION_CONTRACT.md).

    python scripts/check_board.py board/H_13_0.json board/O_8_1.json ...   # any recordings, for debugging
    python scripts/check_board.py --complete board/*.json                  # all 60, each with its fold model

Each file holds one recording, as the board wrote it (the format is in docs/DECISION_CONTRACT.md, section 6):
the model it used, and for each of the 9 windows its features and P(fault), optionally with the PCM and filter
summaries and the model stages computed from the golden features. The five golden recordings are checked stage
by stage; the others by features, probabilities and the decision (reference/a5/window_features.csv and
decisions.csv). Every check prints as a fraction of its tolerance: 1.0 is at the limit, above it fails.

A file that doesn't follow the format is REJECTED before any check: each window needs its ID (1-9, each once),
25 finite features and one P(fault) between 0 and 1; the score must be a finite number between 0 and 1 and the
decision 0 or 1; optional stages must have the right shape and finite values. The board's own numbers must also
agree with each other: its score is the mean of its window probabilities (within SCORE_ROUNDING) and its
decision is score >= 0.5.

A recording checked with models/all_bearings.json is compared on that model's training data: a match shows that
the board computes what the PC computes, not how accurate the model is. Only --complete, with every recording
checked against its held-out fold model, reproduces the leave-one-bearing-out result.
Exits with status 1 if anything is rejected or fails.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bearing_audio import reference as R           # noqa: E402
from bearing_audio.features import FEATURE_NAMES   # noqa: E402

REF = ROOT / "reference" / "a5"
N_WINDOWS = 9
ALL_BEARINGS = "models/all_bearings.json"
SCORE_ROUNDING = 1e-3   # the board's score against the mean of its own window probabilities (allows printing
                        # them to three decimals); the score against the reference keeps its 0.02 tolerance


class BoardFileError(ValueError):
    """A board file that doesn't follow docs/DECISION_CONTRACT.md, section 6. It is rejected, not checked."""


# --------------------------------------------------------------------------------------------- validation

def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _is_integer(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _numbers(v, n: int, what: str) -> None:
    if not isinstance(v, list) or len(v) != n:
        got = f"{len(v)} values" if isinstance(v, list) else repr(v)[:40]
        raise BoardFileError(f"{what}: expected {n} finite numbers, got {got}")
    bad = [j for j, x in enumerate(v) if not _is_number(x)]
    if bad:
        raise BoardFileError(f"{what}: {len(bad)} of {n} values are not finite numbers (first at index {bad[0]}: "
                             f"{v[bad[0]]!r})")


def _probability(v, what: str) -> None:
    if not (_is_number(v) and 0.0 <= v <= 1.0):
        raise BoardFileError(f"{what}: expected one finite number between 0 and 1, got {v!r}")


def _by_window(items, what: str, all_nine: bool) -> dict[int, dict]:
    """Entries by window ID. IDs are integers 1-9, each at most once; with all_nine, each exactly once."""
    if not (isinstance(items, list) and all(isinstance(w, dict) for w in items)):
        raise BoardFileError(f"{what}: expected a list of objects")
    ids = [w.get("window") for w in items]
    bad = [i for i in ids if not (_is_integer(i) and 1 <= i <= N_WINDOWS)]
    if bad:
        raise BoardFileError(f"{what}: window IDs must be integers 1-{N_WINDOWS}, got {bad[:3]}")
    if len(set(ids)) != len(ids):
        raise BoardFileError(f"{what}: window IDs repeated: {sorted({i for i in ids if ids.count(i) > 1})}")
    if all_nine and len(ids) != N_WINDOWS:
        raise BoardFileError(f"{what}: expected windows 1-{N_WINDOWS} once each, got {sorted(ids)}")
    return {w["window"]: w for w in items}


def _signal_summary(v, what: str) -> None:
    if not isinstance(v, dict):
        raise BoardFileError(f"{what}: expected a summary object with peak, rms, mean and probes")
    for k in ("peak", "rms", "mean"):
        if not _is_number(v.get(k)):
            raise BoardFileError(f"{what}.{k}: expected a finite number, got {v.get(k)!r}")
    _numbers(v.get("probes"), len(R.PROBES), f"{what}.probes")


def _pcm_summary(v, what: str) -> None:
    ok = (isinstance(v, dict) and isinstance(v.get("sha256"), str)
          and all(_is_integer(v.get(k)) for k in ("sum", "min", "max"))
          and isinstance(v.get("probes"), list) and len(v["probes"]) == len(R.PROBES)
          and all(_is_integer(x) for x in v["probes"]))
    if not ok:
        raise BoardFileError(f"{what}: expected sha256, integer sum, min and max, and {len(R.PROBES)} integer probes")


def validate(board, decisions: dict, golden: dict | None) -> None:
    """Raise BoardFileError unless the board file has everything the checks need, in the right shape."""
    model = board.get("model")
    lobo = f"models/{decisions[board['recording']]['fold_model']}.json"
    if model not in (lobo, ALL_BEARINGS):
        raise BoardFileError(f"model {model!r} is neither its fold model ({lobo}) nor {ALL_BEARINGS}")
    for i, w in _by_window(board.get("windows"), "windows", all_nine=True).items():
        if "features" not in w or "probability" not in w:
            raise BoardFileError(f"window {i}: features and probability are required")
        _numbers(w["features"], len(FEATURE_NAMES), f"window {i} features")
        _probability(w["probability"], f"window {i} probability")
        if "pcm" in w:
            _pcm_summary(w["pcm"], f"window {i} pcm")
        for s in R.SIGNAL_TOLERANCE:
            if s in w:
                _signal_summary(w[s], f"window {i} {s}")
    if "model_from_golden" in board:
        mfg = _by_window(board["model_from_golden"], "model_from_golden", all_nine=False)
        if golden:
            for i, m in mfg.items():
                for s in ("standardized", "pca", "hidden"):
                    _numbers(m.get(s), len(golden["windows"][i - 1][s]), f"model_from_golden window {i} {s}")
                _probability(m.get("probability"), f"model_from_golden window {i} probability")
    _probability(board.get("score"), "score")
    if board.get("decision") not in (0, 1) or isinstance(board.get("decision"), bool):
        raise BoardFileError(f"decision: expected 0 or 1, got {board.get('decision')!r}")


# --------------------------------------------------------------------------------------------- checks

def expected_windows(name: str, model: str, golden: dict | None, table: dict) -> list[dict]:
    """The reference record of each window: golden stages when available, else features and P(fault)."""
    lobo = f"models/{table['decisions'][name]['fold_model']}.json"
    out = []
    for i in range(N_WINDOWS):
        if golden:
            w = dict(golden["windows"][i])
            if model == ALL_BEARINGS:
                w["probability"] = golden["all_bearings"]["probabilities"][i]
        else:
            row = table["windows"][(name, i + 1)]
            w = {"features": [float(row[n]) for n in FEATURE_NAMES], "lag_gap": float(row["lag_gap"]),
                 "probability": float(row["p_lobo" if model == lobo else "p_all_bearings"])}
        out.append(w)
    return out


def check_board(path: Path, spread, table: dict) -> tuple[bool, list[str], dict | None]:
    """Check one board file. Returns whether it passed, the lines to print, and the board (None if rejected)."""
    try:
        board = json.loads(Path(path).read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
        return False, [f"{path}: REJECTED: not a readable JSON file ({e})"], None
    name = board.get("recording") if isinstance(board, dict) else None
    if not (isinstance(name, str) and name in table["decisions"]):
        return False, [f"{path}: REJECTED: unknown or missing recording {name!r}"], None
    gpath = REF / "golden" / f"{name}.json"
    golden = json.loads(gpath.read_text()) if gpath.exists() else None
    try:
        validate(board, table["decisions"], golden)
    except BoardFileError as e:
        return False, [f"{name} ({path}): REJECTED: {e}"], None

    model = board["model"]
    windows = _by_window(board["windows"], "windows", all_nine=True)
    expected = expected_windows(name, model, golden, table)
    results: dict[str, float] = {}

    def note(check, value):
        value = float(value)
        value = value if not math.isnan(value) else float("inf")
        results[check] = max(results.get(check, 0.0), value)

    for i in range(1, N_WINDOWS + 1):                     # matched by window ID, not by position
        bw = windows[i]
        stages = {k: v for k, v in bw.items()
                  if k in ("features", "probability") or (golden and k in ("pcm", *R.SIGNAL_TOLERANCE))}
        for c, v in R.check_window(stages, expected[i - 1], spread).items():
            note(f"window {i} {c}", v)
    skipped = []
    if board.get("model_from_golden"):
        if golden and model == golden["model"]:
            for i, mw in _by_window(board["model_from_golden"], "model_from_golden", all_nine=False).items():
                for c, v in R.check_model(mw, golden["windows"][i - 1]).items():
                    note(f"window {i} model {c}", v)
        else:
            skipped.append("model_from_golden not checked: golden stages exist only for the five golden "
                           "recordings with their fold models")

    probs = [windows[i]["probability"] for i in range(1, N_WINDOWS + 1)]
    score, decision = float(board["score"]), board["decision"]
    note("recording score against the mean of its window probabilities",
         abs(score - float(np.mean(probs))) / SCORE_ROUNDING)
    note("recording decision against its score and the 0.5 threshold",
         0.0 if decision == int(score >= R.THRESHOLD) else float("inf"))
    dec = table["decisions"][name]
    held_out = model != ALL_BEARINGS
    ref_score = float(dec["lobo_score" if held_out else "all_bearings_score"])
    ref_decision = int(dec["lobo_decision" if held_out else "all_bearings_decision"])
    note("recording score", abs(score - ref_score) / R.PROBABILITY_TOLERANCE["recording"])
    note("recording decision", 0.0 if decision == ref_decision else float("inf"))

    failed = sorted(((v, c) for c, v in results.items() if v > 1.0), reverse=True)
    worst_value, worst = max((v, c) for c, v in results.items())
    kind = "held-out fold model" if held_out else "all-bearings model: parity on its training data, not accuracy"
    lines = [f"{name} ({model}; {kind}): {'FAIL' if failed else 'PASS'}; {len(results)} checks, worst {worst} "
             f"at {worst_value:.3g} of its tolerance; decision {decision} (reference {ref_decision})"]
    lines += [f"    {c}: {v:.3g}" for v, c in failed[:20]]
    if len(failed) > 20:
        lines.append(f"    ... and {len(failed) - 20} more")
    lines += [f"    note: {s}" for s in skipped]
    return not failed, lines, board


def check_file(path: Path, spread, table: dict) -> tuple[bool, list[str]]:
    """Check one board file: whether it passed, and the lines to print."""
    ok, lines, _ = check_board(path, spread, table)
    return ok, lines


def check_complete(boards: list[dict | None], table: dict) -> tuple[bool, list[str]]:
    """With all 60 recordings, each once with its fold model, the board's decisions are a held-out result."""
    names = [b["recording"] for b in boards if b]
    problems = []
    if len(names) != len(boards):
        problems.append(f"{len(boards) - len(names)} file(s) rejected")
    missing = sorted(set(table["decisions"]) - set(names))
    repeated = sorted({n for n in names if names.count(n) > 1})
    not_held_out = sorted(b["recording"] for b in boards if b and b["model"] == ALL_BEARINGS)
    if missing:
        problems.append(f"{len(missing)} recording(s) missing: {', '.join(missing[:5])}{' ...' if len(missing) > 5 else ''}")
    if repeated:
        problems.append(f"recording(s) given more than once: {', '.join(repeated)}")
    if not_held_out:
        problems.append(f"{len(not_held_out)} recording(s) checked with the all-bearings model instead of their "
                        f"fold model: {', '.join(not_held_out[:5])}")
    if problems:
        return False, ["complete evaluation: INCOMPLETE; " + "; ".join(problems)]
    truth = {n: int(d["truth"]) for n, d in table["decisions"].items()}
    pred = {b["recording"]: b["decision"] for b in boards}
    tpr = np.mean([pred[n] == 1 for n in truth if truth[n] == 1])
    tnr = np.mean([pred[n] == 0 for n in truth if truth[n] == 0])
    correct = sum(pred[n] == truth[n] for n in truth)
    return True, [f"complete evaluation: all {len(truth)} recordings with their held-out fold models; "
                  f"{correct} of {len(truth)} correct, balanced accuracy {(tpr + tnr) / 2:.4f}. If every file "
                  "passed, this reproduces the PC's leave-one-bearing-out result; it is not a new estimate."]


def load_table() -> dict:
    with open(REF / "decisions.csv", newline="") as f:
        decisions = {row["recording"]: row for row in csv.DictReader(f)}
    with open(REF / "window_features.csv", newline="") as f:
        windows = {(row["recording"], int(row["window"])): row for row in csv.DictReader(f)}
    return {"decisions": decisions, "windows": windows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--complete", action="store_true",
                    help="also require all 60 recordings, each once, with its held-out fold model")
    a = ap.parse_args()
    spread = np.asarray(json.loads((REF / "contract.json").read_text())["features"]["spread"])
    table = load_table()
    ok, boards = True, []
    for path in a.files:
        passed, lines, board = check_board(path, spread, table)
        ok &= passed
        boards.append(board)
        print("\n".join(lines))
    if a.complete:
        passed, lines = check_complete(boards, table)
        ok &= passed
        print("\n".join(lines))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
