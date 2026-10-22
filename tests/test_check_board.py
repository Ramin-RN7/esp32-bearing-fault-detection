"""scripts/check_board.py (docs/DECISION_CONTRACT.md, section 6): malformed board files are rejected before any
check, non-finite values fail, windows are matched by their IDs, the board's own score and decision must agree
with its window probabilities, and a complete run needs all 60 recordings with their fold models."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from bearing_audio import reference as R
from bearing_audio.features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference" / "a5"


def _load_check_board():
    spec = importlib.util.spec_from_file_location("check_board", ROOT / "scripts" / "check_board.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CB = _load_check_board()
TABLE = CB.load_table()
SPREAD = np.asarray(json.loads((REF / "contract.json").read_text())["features"]["spread"])


def reference_board(name: str, fold: bool = True) -> dict:
    """What an exact port writes for one recording: the frozen features and probabilities, no optional stages."""
    d = TABLE["decisions"][name]
    p = "p_lobo" if fold else "p_all_bearings"
    windows = [{"window": i, "features": [float(TABLE["windows"][(name, i)][n]) for n in FEATURE_NAMES],
                "probability": float(TABLE["windows"][(name, i)][p])} for i in range(1, 10)]
    return {"recording": name, "model": f"models/{d['fold_model']}.json" if fold else CB.ALL_BEARINGS,
            "windows": windows, "score": float(d["lobo_score" if fold else "all_bearings_score"]),
            "decision": int(d["lobo_decision" if fold else "all_bearings_decision"])}


def run(tmp_path: Path, board: dict, file: str = "board.json") -> tuple[bool, str]:
    path = tmp_path / file
    path.write_text(json.dumps(board))                 # writes NaN and Infinity as Python's json reads them
    ok, lines = CB.check_file(path, SPREAD, TABLE)
    return ok, "\n".join(lines)


# --------------------------------------------------------------------------------------------- valid input

@pytest.mark.parametrize("name,fold", [("O_8_1", True), ("B_11_2", True), ("H_13_0", False)])
def test_exact_values_pass(tmp_path, name, fold):
    ok, out = run(tmp_path, reference_board(name, fold))
    assert ok and "REJECTED" not in out, out
    assert ("held-out fold model" in out) == fold
    assert ("not accuracy" in out) == (not fold)


def test_windows_are_matched_by_id_not_by_position(tmp_path):
    board = reference_board("O_8_1")
    board["windows"].reverse()
    ok, out = run(tmp_path, board)
    assert ok, out
    w1, w2 = board["windows"][-1], board["windows"][-2]          # windows 1 and 2 with their IDs swapped
    w1["window"], w2["window"] = 2, 1
    ok, out = run(tmp_path, board)
    assert not ok and "REJECTED" not in out and ("window 1 " in out or "window 2 " in out), out


# --------------------------------------------------------------------------------------------- the three cases
# reproduced by GPT's review (22 October 2026): all three were accepted as PASS before the fix

def _ids_only(b):
    b["windows"] = [{"window": w["window"]} for w in b["windows"]]


def _nan_features_and_probabilities(b):
    for w in b["windows"]:
        w["features"] = [float("nan")] * 24 + w["features"][-1:]
        w["probability"] = float("nan")
    b["score"] = float("nan")


def _ids_999(b):
    for w in b["windows"]:
        w["window"] = 999


@pytest.mark.parametrize("mutate", [_ids_only, _nan_features_and_probabilities, _ids_999])
def test_reviewed_cases_are_rejected(tmp_path, mutate):
    board = reference_board("O_8_1")
    mutate(board)
    ok, out = run(tmp_path, board)
    assert not ok and "REJECTED" in out, out


# --------------------------------------------------------------------------------------------- other malformed input

def _set(path, value):
    """A mutation that sets board[path[0]][path[1]]... to value."""
    def mutate(b):
        target = b
        for k in path[:-1]:
            target = target[k]
        target[path[-1]] = value
    return mutate


def _drop(window, key):
    return lambda b: b["windows"][window].pop(key)


MALFORMED = {
    "missing probability": _drop(4, "probability"),
    "missing features": _drop(0, "features"),
    "24 features": lambda b: b["windows"][2]["features"].pop(),
    "infinite feature": _set(("windows", 1, "features", 3), float("inf")),
    "text feature": _set(("windows", 1, "features", 3), "0.1"),
    "probability as a list": lambda b: b["windows"][3].update(probability=[b["windows"][3]["probability"]]),
    "probability above 1": _set(("windows", 3, "probability"), 1.5),
    "probability as text": _set(("windows", 3, "probability"), "0.3"),
    "repeated window ID": _set(("windows", 8, "window"), 8),
    "window ID 0": _set(("windows", 0, "window"), 0),
    "eight windows": lambda b: b["windows"].pop(),
    "score not a number": _set(("score",), float("nan")),
    "score above 1": _set(("score",), 1.2),
    "score as a list": lambda b: b.update(score=[b["score"]]),
    "decision true": _set(("decision",), True),
    "decision 2": _set(("decision",), 2),
    "unknown recording": _set(("recording",), "X_99_9"),
    "another fold's model": _set(("model",), "models/lobo_bearing_01.json"),
}


@pytest.mark.parametrize("case", MALFORMED)
def test_malformed_files_are_rejected(tmp_path, case):
    board = reference_board("O_8_1")
    MALFORMED[case](board)
    ok, out = run(tmp_path, board)
    assert not ok and "REJECTED" in out, (case, out)


def test_malformed_optional_stages_are_rejected(tmp_path):
    golden = json.loads((REF / "golden" / "B_11_2.json").read_text())
    board = reference_board("B_11_2")
    board["windows"][0]["notch"] = dict(golden["windows"][0]["notch"], peak=float("nan"))
    ok, out = run(tmp_path, board)
    assert not ok and "REJECTED" in out and "notch" in out, out
    board = reference_board("B_11_2")
    board["model_from_golden"] = [{"window": 1, **{s: golden["windows"][0][s] for s in ("standardized", "hidden")},
                                   "pca": golden["windows"][0]["pca"][:3], "probability": 0.5}]
    ok, out = run(tmp_path, board)
    assert not ok and "REJECTED" in out and "pca" in out, out


def test_unreadable_file_is_rejected(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"recording": "O_8_1", ')
    ok, lines = CB.check_file(path, SPREAD, TABLE)
    assert not ok and "REJECTED" in lines[0]


# --------------------------------------------------------------------------------------------- the board's own numbers

def test_score_must_be_the_mean_of_the_window_probabilities(tmp_path):
    board = reference_board("O_8_1")
    board["score"] += -0.015 if board["score"] > 0.5 else 0.015   # within the 0.02 reference tolerance: passed before
    ok, out = run(tmp_path, board)
    assert not ok and "mean of its window probabilities" in out, out


def test_decision_must_follow_the_score_and_threshold(tmp_path):
    board = reference_board("O_8_1")
    board["decision"] = 1 - board["decision"]
    ok, out = run(tmp_path, board)
    assert not ok and "its score and the 0.5 threshold" in out, out


# --------------------------------------------------------------------------------------------- genuine mismatches

def test_numerical_mismatches_fail(tmp_path):
    board = reference_board("O_8_1")
    p = board["windows"][2]["probability"]
    board["windows"][2]["probability"] = p - 0.1 if p > 0.5 else p + 0.1
    board["score"] = float(np.mean([w["probability"] for w in board["windows"]]))
    ok, out = run(tmp_path, board)
    assert not ok and "window 3 probability" in out, out

    board = reference_board("O_8_1")
    j = FEATURE_NAMES.index("rms")
    board["windows"][4]["features"][j] += 10 * R.FEATURE_TOLERANCE["rms"] * SPREAD[j]
    ok, out = run(tmp_path, board)
    assert not ok and "window 5 rms" in out, out


# --------------------------------------------------------------------------------------------- reference.py checks

def test_reference_checks_fail_on_non_finite_values_and_wrong_shapes():
    g = json.loads((REF / "golden" / "B_11_2.json").read_text())["windows"][0]
    feats = list(g["features"])
    assert max(R.check_features(feats, g["features"], SPREAD).values()) == 0
    feats[0] = float("nan")
    assert R.check_features(feats, g["features"], SPREAD)["mean"] == float("inf")
    lag = list(g["features"])
    lag[-1] = float("nan")
    assert R.check_features(lag, g["features"], SPREAD, lag_gap_value=0.0)["autocorr_lag"] == float("inf")
    with pytest.raises(ValueError):
        R.check_features(g["features"][:24], g["features"], SPREAD)
    assert R.check_window({"probability": float("nan")}, g, SPREAD)["probability"] == float("inf")
    notch = dict(g["notch"], probes=[float("nan")] + g["notch"]["probes"][1:])
    assert R.check_window({"notch": notch}, g, SPREAD)["notch"] == float("inf")
    stages = {s: list(g[s]) for s in ("standardized", "pca", "hidden")}
    stages["probability"] = g["probability"]
    assert max(R.check_model(stages, g).values()) == 0
    stages["hidden"][0] = float("nan")
    assert R.check_model(stages, g)["hidden"] == float("inf")
    with pytest.raises(ValueError):
        R.check_model({**stages, "pca": g["pca"][:3]}, g)


# --------------------------------------------------------------------------------------------- complete evaluation

def test_complete_needs_all_60_recordings_with_their_fold_models(tmp_path):
    names = sorted(TABLE["decisions"])
    boards = [reference_board(n) for n in names]
    ok, lines = CB.check_complete(boards, TABLE)
    assert ok and "58 of 60 correct, balanced accuracy 0.9625" in lines[0], lines
    for broken in (boards[:-1],                                            # one missing
                   boards[:-1] + [boards[0]],                              # one twice, one missing
                   boards[:-1] + [reference_board(names[-1], fold=False)],  # all-bearings model
                   boards[:-1] + [None]):                                  # one rejected
        ok, lines = CB.check_complete(broken, TABLE)
        assert not ok and "INCOMPLETE" in lines[0], lines


def test_command_line_exit_status(tmp_path):
    files = []
    for n in sorted(TABLE["decisions"]):
        files.append(tmp_path / f"{n}.json")
        files[-1].write_text(json.dumps(reference_board(n)))
    script = [sys.executable, str(ROOT / "scripts" / "check_board.py")]
    done = subprocess.run(script + ["--complete", *map(str, files)], capture_output=True, text=True)
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    assert "balanced accuracy 0.9625" in done.stdout
    done = subprocess.run(script + ["--complete", *map(str, files[:-1])], capture_output=True, text=True)
    assert done.returncode == 1 and "INCOMPLETE" in done.stdout
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**reference_board("O_8_1"), "windows": []}))
    done = subprocess.run(script + [str(bad)], capture_output=True, text=True)
    assert done.returncode == 1 and "REJECTED" in done.stdout
