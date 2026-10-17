"""The rebuilt ablation pipelines: configuration parsing, fold integrity and agreement with the archived runs."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bearing_audio import ablation
from bearing_audio.data import list_recordings, load_original_split
from bearing_audio.evaluation import lobo_folds

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "ablation_lobo.json"


def test_configurations_cover_the_ablation():
    cfgs = ablation.configs()
    ids = [c.id for c in cfgs]
    assert len(ids) == len(set(ids)) == 86
    assert sum(not c.base for c in cfgs) == 82                     # window-level configurations
    assert {c.id for c in cfgs if c.base} == set(ablation.AGGREGATION_VARIANTS)
    assert all((ROOT / c.run / "code.py").exists() for c in cfgs)
    by_id = {c.id: c for c in cfgs}
    assert by_id["A14-GMM"].outlier == "mahalanobis" and by_id["A17-GMM"].outlier_pct == pytest.approx(0.10)
    assert by_id["E3-GMM"].outlier == "gmm-trim" and by_id["A15-GMM"].threshold == "fixed-p5"
    assert by_id["A44-ET"].trees == (15, 6) and by_id["A44-ET"].window_rule == "proba"
    assert by_id["A18-MLP"].scaler == "robust" and by_id["A26-GMM"].pca == 0


def test_nested_inner_folds_never_see_the_outer_bearing():
    recordings = {r.name: r for r in list_recordings()}
    for outer in lobo_folds(recordings):
        b = outer["test_bearing"]
        inner = lobo_folds(recordings, exclude_bearings=(b,))
        assert len(inner) == 19
        for f in inner:
            names = f["train"] + f["val"] + f["test"]
            assert all(recordings[n].bearing != b for n in names)
            assert not ({recordings[n].bearing for n in f["test"]} & {recordings[n].bearing for n in f["train"] + f["val"]})


def test_archived_feature_code_reproduces_the_a16_features_and_counts(table):
    """Runs A16's own code.py on the exported segments; features and confusion counts must match exactly."""
    cfgs = [c for c in ablation.configs() if c.id in ("A16-GMM", "A16-MLP")]
    feats = ablation.ablation_features(list_recordings(), cfgs)
    assert feats["A16-MLP"].dtype == table.a16.dtype and np.array_equal(feats["A16-MLP"], table.a16)
    split = load_original_split()
    pick = lambda s: [n for n in sorted(table.recordings) if split[n] == s]
    out = ablation.run_split(cfgs, feats, table, pick("train"), pick("val"), pick("test"))
    for c in cfgs:
        got = ablation.pooled([out], c.id)["window"]
        assert got == {k: c.archived[k] for k in ("tp", "fp", "tn", "fn")}, c.id


@pytest.mark.skipif(not RESULTS.exists(), reason="results/ablation_lobo.json not generated")
def test_published_ablation_results_agree_with_the_protocol_comparison():
    """The two evaluation scripts share no model code paths beyond the package; their LOBO decisions must agree."""
    ab = json.loads(RESULTS.read_text())
    pc = json.loads((ROOT / "results" / "protocol_comparison.json").read_text())["leave_one_bearing_out"]
    for a, p in (("A16-MLP", "a16_mlp"), ("A8-GMM", "a8_gmm"), ("BASELINE", "baseline_rms_kurtosis")):
        mine = {(x["recording"], x["pred"]) for x in ab["lobo"][a]["predictions"]}
        theirs = {(x["recording"], x["pred"]) for x in pc[p]["predictions"]}
        assert mine == theirs, a


@pytest.mark.skipif(not RESULTS.exists(), reason="results/ablation_lobo.json not generated")
def test_published_ablation_report_matches_its_json():
    spec = importlib.util.spec_from_file_location("evaluate_ablation", ROOT / "scripts" / "evaluate_ablation.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    res = json.loads(RESULTS.read_text())
    assert mod.markdown(res) == (ROOT / "results" / "ablation_lobo.md").read_text()
    nested = res["nested"]
    assert len(nested["outer_folds"]) == 20 and len(nested["predictions"]) == 60
    assert sorted(p["recording"] for p in nested["predictions"]) == sorted(r.name for r in list_recordings())
