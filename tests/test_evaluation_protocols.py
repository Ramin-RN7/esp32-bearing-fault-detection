"""Fold integrity of the follow-up evaluations, and a one-fold check of the published leave-one-bearing-out results."""
import json

import numpy as np

from bearing_audio.data import list_recordings
from bearing_audio.evaluation import (evaluate_split, family_balanced_validation, fault_family, lobo_folds,
                                      matched_splits)

RECS = {r.name: r for r in list_recordings()}
BALL_FAULT_BEARINGS = (11, 12, 13, 14, 15)


def _bearings(names):
    return {RECS[n].bearing for n in names}


def test_lobo_holds_out_every_recording_once_and_keeps_bearings_apart():
    for exclude in ((), BALL_FAULT_BEARINGS):
        folds = lobo_folds(RECS, exclude)
        held_out = [n for f in folds for n in f["test"]]
        assert sorted(held_out) == sorted(n for n, r in RECS.items() if r.bearing not in exclude)
        assert len(held_out) == len(set(held_out))
        n_bearings = len({r.bearing for r in RECS.values()}) - len(exclude)
        for f in folds:
            tr, va, te = _bearings(f["train"]), _bearings(f["val"]), _bearings(f["test"])
            assert te == {f["test_bearing"]} and va == set(f["validation_bearings"]) and len(va) == 3
            assert not (tr & va or tr & te or va & te)
            assert len(tr) == n_bearings - 4 and not (tr | va | te) & set(exclude)


def test_alternative_validation_selections_span_three_fault_types():
    family = fault_family(RECS)
    for seed in range(10):
        for f in lobo_folds(RECS, choose_validation=family_balanced_validation(RECS, seed)):
            assert len({family[b] for b in f["validation_bearings"]}) == 3
            assert f["test_bearing"] not in f["validation_bearings"]


def test_bearing_level_repeated_splits_never_share_a_bearing():
    class Recordings:
        recordings = RECS
    for seed in range(200):
        _, (train, val, test) = matched_splits(Recordings, seed)
        tr, va, te = _bearings(train), _bearings(val), _bearings(test)
        assert not (tr & va or tr & te or va & te)


def test_one_lobo_fold_matches_the_published_results(table, root):
    published = json.loads((root / "results" / "protocol_comparison.json").read_text())["leave_one_bearing_out"]
    fold = next(f for f in lobo_folds(table.recordings) if f["test_bearing"] == 13)
    res = evaluate_split(table, fold["train"], fold["val"], fold["test"])
    for k, r in res.items():
        expected = {p["recording"]: p for p in published[k]["predictions"] if p["recording"] in fold["test"]}
        assert {n: expected[n]["pred"] for n in r["test"]} == dict(zip(r["test"], r["pred"]))
        if all("score" in p for p in expected.values()):
            np.testing.assert_allclose(r["score"], [expected[n]["score"] for n in r["test"]], rtol=1e-4, atol=1e-6)
