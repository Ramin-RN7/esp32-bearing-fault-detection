#!/usr/bin/env python3
"""Re-evaluate every thesis ablation configuration on unseen bearings (2026 follow-up, not part of the thesis).

    python scripts/evaluate_ablation.py                 # features, original split, LOBO and nested selection
    python scripts/evaluate_ablation.py --stage lobo    # stop after leave-one-bearing-out

Steps:
1. Rebuild the 82 window-level configurations from their run records (bearing_audio.ablation) and check each
   against its archived confusion counts on the original split. A40/A41 (majority and median aggregation) are
   added for A8 GMM and A16 MLP, giving 86 pipelines.
2. Leave-one-bearing-out (protocol v1, as in evaluate_protocols.py) for all 86 pipelines plus the RMS + kurtosis
   reference, with bearing-bootstrap intervals.
3. Nested selection: in every outer fold, choose the pipeline by leave-one-bearing-out on the 19 remaining
   bearings only, then test the chosen pipeline on the held-out bearing. This scores the procedure "pick the best
   of 86" without letting the held-out bearing influence the choice.

Writes results/ablation_lobo.json, results/ablation_lobo.csv and results/ablation_lobo.md. Features are cached in
artifacts/ablation_features/ (git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from bearing_audio import __version__                                                   # noqa: E402
from bearing_audio.ablation import (BASELINE, NOT_SEPARATE, ablation_features, balanced_accuracy,  # noqa: E402
                                    bootstrap, configs, fault_f1, pooled, run_split, select)
from bearing_audio.data import list_recordings, load_original_split                    # noqa: E402
from bearing_audio.evaluation import lobo_folds                                          # noqa: E402
from bearing_audio.features import FeatureTable                                          # noqa: E402

_STATE: dict = {}          # per process; worker processes fill it from the caches in _init_worker


def _load_state(workers: int = 1, progress=None):
    recordings = list_recordings()
    table = FeatureTable(recordings, cache=ROOT / "artifacts" / "features.npz")
    cfgs = configs()
    all_cfgs = cfgs + [BASELINE]
    feats = ablation_features(recordings, all_cfgs, cache_dir=ROOT / "artifacts" / "ablation_features",
                              workers=workers, progress=progress)
    _STATE.update(cfgs=cfgs, all_cfgs=all_cfgs, feats=feats, table=table)


def _init_worker():
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    _load_state()


def _pool(workers: int) -> ProcessPoolExecutor:
    # "spawn", not fork: a process forked after numpy/scikit-learn have started their thread pools can deadlock.
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"), initializer=_init_worker)


def _lobo_fold(fold):
    s = _STATE
    return run_split(s["all_cfgs"], s["feats"], s["table"], fold["train"], fold["val"], fold["test"])


def _nested_outer(test_bearing):
    s = _STATE
    inner = [run_split(s["cfgs"], s["feats"], s["table"], f["train"], f["val"], f["test"])
             for f in lobo_folds(s["table"].recordings, exclude_bearings=(test_bearing,))]
    chosen, scores = select(s["cfgs"], inner)
    best = scores[chosen]["recording_ba"]
    tied = sorted(k for k, v in scores.items() if v["recording_ba"] == best)
    return {"test_bearing": int(test_bearing), "chosen": chosen, "inner_recording_ba": best,
            "inner_window_ba": scores[chosen]["window_ba"], "tied_on_recording_ba": tied,
            "inner_scores": scores}


def family(cfg, params) -> str:
    feats = str(params.get("features") or "")
    pre = params.get("preprocess") or []
    pre = " ".join(pre) if isinstance(pre, list) else str(pre)
    if "logmel" in feats or "mfcc" in feats:
        return "Log-mel / MFCC summaries"
    if "hilbert" in pre:
        return f"25 features on the envelope, {int(params.get('sr', 16000)) // 1000} kHz"
    if "bandpass_1000-5000Hz" in pre:
        return "25 features after a 1–5 kHz band-pass"
    if not pre:
        return "25 features on raw audio"
    return "25 features after HPF, pre-emphasis, notch or whitening"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", choices=("reproduce", "lobo", "nested"), default="nested")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    t0 = time.time()

    _load_state(args.workers, progress=lambda i, n: print(f"  features {i}/{n} ({time.time() - t0:.0f} s)", flush=True))
    cfgs, all_cfgs, feats, table = (_STATE[k] for k in ("cfgs", "all_cfgs", "feats", "table"))
    print(f"{len(cfgs)} pipelines, features ready ({time.time() - t0:.0f} s)", flush=True)

    params = {c.id: json.loads((ROOT / c.run / "params.json").read_text()) for c in cfgs}
    res = {"version": __version__, "n_pipelines": len(cfgs), "not_separate_pipelines": NOT_SEPARATE,
           "pipelines": {c.id: {"description": c.description, "model": c.model, "scaler": c.scaler, "pca": c.pca,
                                "threshold": c.threshold if c.model == "GMM" else "probability 0.5",
                                "outlier_removal": c.outlier if c.outlier == "none" else f"{c.outlier} {c.outlier_pct:g}",
                                "aggregation": c.aggregation, "family": family(c, params[c.id]), "run": c.run}
                         for c in cfgs}}

    # 1. original split
    split = load_original_split()
    pick = lambda s: [n for n in sorted(table.recordings) if split[n] == s]
    orig = run_split(all_cfgs, feats, table, pick("train"), pick("val"), pick("test"))
    repro = {}
    for c in cfgs:
        level = c.archived["level"]
        got = pooled([orig], c.id)[level]
        want = {k: c.archived[k] for k in ("tp", "fp", "tn", "fn")}
        repro[c.id] = {"level": level, "archived": want, "rerun": got, "match": got == want,
                       "original_recording": pooled([orig], c.id)["recording"]}
    res["original_split"] = repro
    n_match = sum(v["match"] for v in repro.values())
    print(f"original split: {n_match}/{len(repro)} pipelines reproduce their archived counts", flush=True)
    for k, v in repro.items():
        if not v["match"]:
            print(f"  {k}: archived {v['archived']} rerun {v['rerun']}")
    if args.stage == "reproduce":
        _write(res, args.out, partial=True)
        return

    # 2. leave-one-bearing-out
    folds = lobo_folds(table.recordings)
    with _pool(args.workers) as ex:
        lobo_res = list(ex.map(_lobo_fold, folds))
    print(f"leave-one-bearing-out done ({time.time() - t0:.0f} s)", flush=True)
    preds = {}
    for c in all_cfgs:
        preds[c.id] = [{"recording": n, "test_bearing": f["test_bearing"], "true": t, "pred": p, "score": s}
                       for f, r in zip(folds, lobo_res) for (n, t, p, s) in r[c.id]["recordings"]]
    lobo = {}
    for c in all_cfgs:
        p = pooled(lobo_res, c.id)
        lobo[c.id] = {"recording": p["recording"], "window": p["window"],
                      "balanced_accuracy": balanced_accuracy(p["recording"]), "fault_f1": fault_f1(p["recording"]),
                      "window_balanced_accuracy": balanced_accuracy(p["window"]),
                      "thresholds": [r[c.id]["threshold"] for r in lobo_res], "predictions": preds[c.id]}
    res["lobo"] = lobo
    res["lobo_folds"] = [{"test_bearing": f["test_bearing"], "validation_bearings": f["validation_bearings"]}
                         for f in folds]
    order = sorted(cfgs, key=lambda c: (-lobo[c.id]["balanced_accuracy"], -lobo[c.id]["window_balanced_accuracy"]))
    top = order[0].id
    pairs = [(top, "A16-MLP"), (top, "BASELINE"), ("A16-MLP", "BASELINE"), ("A8-GMM", "BASELINE")]
    pairs = [(a, b) for a, b in pairs if a != b]

    if args.stage == "nested":
        with _pool(args.workers) as ex:
            outer = list(ex.map(_nested_outer, [f["test_bearing"] for f in folds]))
        print(f"nested selection done ({time.time() - t0:.0f} s)", flush=True)
        by_bearing = {o["test_bearing"]: o for o in outer}
        nested_preds = []
        for f in folds:
            chosen = by_bearing[f["test_bearing"]]["chosen"]
            nested_preds += [p for p in preds[chosen] if p["test_bearing"] == f["test_bearing"]]
        preds["NESTED"] = nested_preds
        rec_t = [p["true"] for p in nested_preds]
        rec_p = [p["pred"] for p in nested_preds]
        from bearing_audio.ablation import confusion
        cm = confusion(rec_t, rec_p)
        counts = {}
        for o in outer:
            counts[o["chosen"]] = counts.get(o["chosen"], 0) + 1
        res["nested"] = {"recording": cm, "balanced_accuracy": balanced_accuracy(cm), "fault_f1": fault_f1(cm),
                         "chosen_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
                         "outer_folds": [{k: v for k, v in o.items() if k != "inner_scores"} for o in outer],
                         "predictions": nested_preds,
                         "selection_rule": "highest leave-one-bearing-out recording balanced accuracy on the 19 "
                                           "other bearings; ties to higher window balanced accuracy, then table order"}
        pairs.append(("NESTED", "BASELINE"))
    res["bootstrap"] = bootstrap(preds, table.recordings, pairs=pairs)
    res["runtime_s"] = round(time.time() - t0)
    _write(res, args.out)
    print(f"wrote {args.out / 'ablation_lobo.md'} ({time.time() - t0:.0f} s)")


def _clean(obj):
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return _clean(obj.item())
    return obj


def _write(res, out: Path, partial=False):
    out.mkdir(parents=True, exist_ok=True)
    name = "ablation_lobo_partial.json" if partial else "ablation_lobo.json"
    (out / name).write_text(json.dumps(_clean(res), indent=1, allow_nan=False) + "\n")
    if partial:
        return
    rows = table_rows(res)
    with open(out / "ablation_lobo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (out / "ablation_lobo.md").write_text(markdown(res))


def table_rows(res) -> list[dict]:
    """One row per pipeline, ranked by leave-one-bearing-out balanced accuracy."""
    lobo, ci, orig = res["lobo"], res["bootstrap"]["balanced_accuracy_95"], res["original_split"]
    ids = sorted(res["pipelines"], key=lambda k: (-lobo[k]["balanced_accuracy"], -lobo[k]["window_balanced_accuracy"]))
    rows = []
    for rank, k in enumerate(ids, 1):
        p, l, o = res["pipelines"][k], lobo[k], orig[k]
        rows.append({
            "rank": rank, "id": k, "description": p["description"], "family": p["family"], "model": p["model"],
            "scaler": p["scaler"], "pca": p["pca"], "aggregation": p["aggregation"],
            "reproduces_archived_counts": o["match"],
            "original_split_recording_ba": round(balanced_accuracy(o["original_recording"]), 4),
            "lobo_recording_ba": round(l["balanced_accuracy"], 4),
            "lobo_ba_95_low": round(ci[k][0], 4), "lobo_ba_95_high": round(ci[k][1], 4),
            "lobo_fault_f1": round(l["fault_f1"], 4), "lobo_window_ba": round(l["window_balanced_accuracy"], 4),
            "times_chosen_in_nested_selection": res.get("nested", {}).get("chosen_counts", {}).get(k, 0),
        })
    return rows


NAMES_FOR_REFERENCE = {"A16-MLP": "A16 MLP (thesis best F1)", "A8-GMM": "A8 GMM (ported to the ESP32)",
                       "BASELINE": "Baseline: RMS + kurtosis (2026 reference)"}


def markdown(res) -> str:
    rows = table_rows(res)
    lobo, boot = res["lobo"], res["bootstrap"]
    ci = boot["balanced_accuracy_95"]
    n = res["n_pipelines"]
    repro = res["original_split"]
    n_match = sum(v["match"] for v in repro.values())
    fmt_ci = lambda k: f"{ci[k][0]:.2f}–{ci[k][1]:.2f}"
    lines = [
        "# Every ablation pipeline on unseen bearings",
        "",
        f"Generated by `python scripts/evaluate_ablation.py` (bearing_audio {res['version']}). A 2026 follow-up, not "
        "part of the thesis. Each pipeline is rebuilt from its archived run record: the run's own feature code on the "
        "archived two-second segments, and the model stage from its settings. Decisions are per recording (mean of "
        "the window scores unless the pipeline aggregates otherwise). Balanced accuracy (BA): 0.5 is chance.",
        "",
        "## 1. Rebuilt pipelines",
        "",
        f"{n} pipelines: the 82 window-level configurations and the A40/A41 aggregation variants of A8 GMM and A16 "
        "MLP. The other 7 ablation IDs are not separate pipelines (see the JSON). On the original thesis split, "
        f"**{n_match} of {n}** rebuilt pipelines reproduce their archived confusion counts exactly.",
        "",
    ]
    bad = [k for k, v in repro.items() if not v["match"]]
    if bad:
        lines += ["| Pipeline | Level | Archived (TP/FP/TN/FN) | Rebuilt (TP/FP/TN/FN) |", "|---|---|---|---|"]
        c = lambda d: f"{d['tp']}/{d['fp']}/{d['tn']}/{d['fn']}"
        for k in bad:
            v = repro[k]
            lines.append(f"| {k} | {v['level']} | {c(v['archived'])} | {c(v['rerun'])} |")
        lines.append("")
    best = rows[0]
    lines += [
        "## 2. Leave-one-bearing-out (20 folds, 60 recordings)",
        "",
        "Same folds as `results/protocol_comparison.md`: each bearing is held out once, three other bearings set "
        "the GMM thresholds, the rest train. 95% intervals resample the 20 bearings with the predictions fixed.",
        "",
        "| Rank | Pipeline | Model | Family | BA [95% interval] | Fault F1 | Window BA | Original-split BA |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    shown = rows[:15] + [r for r in rows[15:] if r["id"] in ("A16-MLP", "A8-GMM")]
    for r in shown:
        lines.append(f"| {r['rank']} | {r['id']} | {r['model']} | {r['family']} | {r['lobo_recording_ba']:.3f} "
                     f"[{r['lobo_ba_95_low']:.2f}–{r['lobo_ba_95_high']:.2f}] | {r['lobo_fault_f1']:.3f} "
                     f"| {r['lobo_window_ba']:.3f} | {r['original_split_recording_ba']:.3f} |")
    lines += [f"| – | BASELINE | LR | RMS + kurtosis (reference) | {lobo['BASELINE']['balanced_accuracy']:.3f} "
              f"[{fmt_ci('BASELINE')}] | {lobo['BASELINE']['fault_f1']:.3f} "
              f"| {lobo['BASELINE']['window_balanced_accuracy']:.3f} | – |", ""]
    top_ba = best["lobo_recording_ba"]
    at_top = [r["id"] for r in rows if abs(r["lobo_recording_ba"] - top_ba) < 1e-12]
    above_base = sum(r["lobo_recording_ba"] > lobo["BASELINE"]["balanced_accuracy"] + 1e-12 for r in rows)
    lines += [
        f"The highest leave-one-bearing-out BA is {top_ba:.3f}"
        + (f", reached by {len(at_top)} pipelines ({', '.join(at_top)})" if len(at_top) > 1 else f" ({best['id']})")
        + f". {above_base} of {n} pipelines score above the RMS + kurtosis reference "
        f"({lobo['BASELINE']['balanced_accuracy']:.3f}). All {n} rows are in `ablation_lobo.csv`.",
        "",
        "Paired differences in BA (same resampled bearings for both), 95% interval:",
        "",
    ]
    for pair, (lo, hi) in boot["paired_difference_95"].items():
        lines.append(f"- {pair}: {lo:+.3f} to {hi:+.3f}")
    lines += ["", "### By family", "", "| Family | Model | Pipelines | Best BA | Median BA |",
              "|---|---|---:|---:|---:|"]
    groups = {}
    for r in rows:
        groups.setdefault((r["family"], r["model"]), []).append(r["lobo_recording_ba"])
    for (fam, model), v in sorted(groups.items(), key=lambda kv: -max(kv[1])):
        lines.append(f"| {fam} | {model} | {len(v)} | {max(v):.3f} | {float(np.median(v)):.3f} |")
    if "nested" in res:
        ns = res["nested"]
        chosen = ", ".join(f"{k} ({v})" for k, v in ns["chosen_counts"].items())
        diff = boot["paired_difference_95"].get("NESTED - BASELINE")
        lines += [
            "",
            "## 3. Choosing the pipeline without seeing the held-out bearing (nested selection)",
            "",
            "Picking the top row above uses the same 20 bearings to choose and to score, so its BA is optimistic. "
            "Here each outer fold chooses among all "
            f"{n} pipelines by leave-one-bearing-out on the other 19 bearings only, then tests the chosen pipeline on "
            "the held-out bearing.",
            "",
            f"- **Nested BA: {ns['balanced_accuracy']:.3f} [{fmt_ci('NESTED')}]**, fault F1 {ns['fault_f1']:.3f}, "
            f"against {top_ba:.3f} for the best single pipeline chosen with hindsight.",
            f"- Pipelines chosen across the 20 outer folds: {chosen}.",
        ]
        if diff:
            lines.append(f"- Nested selection minus the RMS + kurtosis reference: {diff[0]:+.3f} to {diff[1]:+.3f}.")
    lines += [
        "",
        "## Limits",
        "",
        "- 20 bearings from one test rig. The intervals leave out refitting variation; differences of a few "
        "recordings between pipelines are within noise.",
        "- Ball-fault recordings were made without load (dataset description), a possible shortcut cue.",
        "- Rebuilt pipelines that do not reproduce their archived counts are still evaluated as specified by their "
        "settings; see section 1.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
