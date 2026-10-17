#!/usr/bin/env python3
"""Quick demo: the best ablation pipeline, A5 MLP, tested on bearings it never trained on.

    python scripts/demo.py          # the summary; --all lists all 60 decisions

Leave-one-bearing-out over the 20 bearings, with the published folds: each round trains on 16 bearings and decides
the 3 recordings of the held-out bearing (the 3 validation bearings of each fold set GMM thresholds and are not
used by the MLP). A recording is called a fault when the mean fault probability of its nine two-second windows is
at least 0.5. Prints the wrong decisions (every decision with --all) and the pooled scores;
results/ablation_lobo.md reports the same run.

A5 MLP was picked as the best of 86 pipelines after seeing all their results, so its score is optimistic. The
estimate that accounts for that choice is nested selection, 0.950 (results/ablation_lobo.md, section 3).
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.exceptions import ConvergenceWarning                                       # noqa: E402

from bearing_audio.ablation import (ablation_features, balanced_accuracy, configs, fault_f1,  # noqa: E402
                                    pooled, run_split)
from bearing_audio.data import list_recordings                                           # noqa: E402
from bearing_audio.evaluation import lobo_folds                                          # noqa: E402
from bearing_audio.features import FeatureTable                                          # noqa: E402

PIPELINE = "A5-MLP"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true", help="print every recording's decision")
    show_all = ap.parse_args().all
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    recordings = list_recordings()
    table = FeatureTable(recordings, cache=ROOT / "artifacts" / "features.npz")
    cfg = [c for c in configs() if c.id == PIPELINE]
    feats = ablation_features(recordings, cfg, cache_dir=ROOT / "artifacts" / "ablation_features")

    print(f"{PIPELINE} (100 Hz high-pass, pre-emphasis, 50 Hz notch, 25 features, z-score, PCA(4), MLP(10))")
    print("Leave-one-bearing-out: 20 folds, 60 recordings, each decided by a model that never heard its bearing\n")
    label = lambda y: "fault" if y else "healthy"
    print("Bearing  Recording  Truth    Decision  Mean P(fault)")
    results = []
    for fold in lobo_folds(table.recordings):
        res = run_split(cfg, feats, table, fold["train"], fold["val"], fold["test"])
        results.append(res)
        for name, true, pred, score in res[PIPELINE]["recordings"]:
            if show_all or pred != true:
                mark = "" if pred == true else "  <- wrong"
                print(f"{fold['test_bearing']:>7}  {name:<9}  {label(true):<7}  {label(pred):<8}  {score:13.3f}{mark}")
    c = pooled(results, PIPELINE)["recording"]
    print(f"\nTP {c['tp']}, FP {c['fp']}, TN {c['tn']}, FN {c['fn']}: "
          f"balanced accuracy {balanced_accuracy(c):.4f}, fault F1 {fault_f1(c):.4f}")


if __name__ == "__main__":
    main()
