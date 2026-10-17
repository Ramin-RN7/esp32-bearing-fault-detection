#!/usr/bin/env python3
"""Reproduce the thesis results on the archived split, then evaluate the same pipelines more strictly.

    python scripts/evaluate_protocols.py                 # full run: 200 matched splits per protocol (~15 min)
    python scripts/evaluate_protocols.py --repeats 20    # quicker

Writes results/protocol_comparison.json, results/protocol_comparison.md and results/repeated_splits.csv.
Window features are cached in artifacts/features.npz (git-ignored). These full evaluations are run by hand;
CI runs the tests, which include a one-fold check of the leave-one-bearing-out predictions published here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# The thesis MLP uses max_iter=500 and often stops before convergence; that is part of the archived configuration.
warnings.filterwarnings("ignore", category=ConvergenceWarning)
BALL_FAULT_BEARINGS = (11, 12, 13, 14, 15)

from bearing_audio import __version__                                        # noqa: E402
from bearing_audio.data import list_recordings, load_original_split           # noqa: E402
from bearing_audio.evaluation import (bearing_bootstrap, fault_family,        # noqa: E402
                                      family_balanced_validation, leave_one_bearing_out, lobo_folds,
                                      matched_splits, original, repeated, summarise_repeated)
from bearing_audio.features import FeatureTable, feature_provenance, file_sha256  # noqa: E402
from bearing_audio.pipelines import PIPELINES                                 # noqa: E402

NAMES = {k: v.name for k, v in PIPELINES.items()}


def provenance(recordings) -> dict:
    """Code version, data and environment behind this run."""
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    audio = hashlib.sha256("".join(f"{r.name}:{file_sha256(r.path)}\n" for r in recordings).encode()).hexdigest()
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "librosa")}
    return {
        "git_commit": git("rev-parse", "HEAD") + ("+uncommitted changes" if dirty else ""),
        "audio_sha256": audio, "n_recordings": len(recordings),
        **feature_provenance(),
        "python": platform.python_version(), "packages": versions,
    }


def clean(obj):
    """NaN (an undefined metric) becomes null, so the JSON is strict."""
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    return obj


def cm(m):
    (tn, fp), (fn, tp) = m["confusion_matrix"]
    return f"TN {tn} · FP {fp} · FN {fn} · TP {tp}"


def num(x, fmt=".2f"):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, fmt)


def lobo_rows(lobo: dict, intervals: dict | None = None) -> list[str]:
    head = "| Pipeline | Confusion | Fault F1 | Balanced accuracy |"
    head += " 95% interval (bearing bootstrap) |" if intervals else ""
    head += " MCC | Specificity | Recall: developing | Recall: faulty |"
    rows = [head, "|---|---|---:|---:|" + ("---:|" if intervals else "") + "---:|---:|---:|---:|"]
    for k, v in lobo.items():
        ci = f" {intervals[k][0]:.2f}–{intervals[k][1]:.2f} |" if intervals else ""
        rows.append(f"| {NAMES[k]} | {cm(v)} | {v['fault_f1']:.3f} | {v['balanced_accuracy']:.3f} |{ci} {v['mcc']:.2f} "
                    f"| {num(v['specificity'])} | {num(v['recall_developing_fault'])} | {num(v['recall_faulty'])} |")
    return rows


def protocol_difference_note(diffs: dict, n: int) -> str:
    """Describe the measured recording-minus-bearing differences, whatever their size."""
    largest = max(round(abs(d), 3) for d in diffs.values())   # as printed in the list above
    if largest < 0.02:
        finding = (f"Over these {n} splits, every mean difference is below 0.02, so no large average difference was "
                   "seen between splitting by recording and by bearing.")
    else:
        finding = (f"Over these {n} splits, the largest mean difference is {largest:.3f}, so splitting by recording "
                   "and by bearing did not give similar averages for every pipeline.")
    return finding + (" The splits reuse the same bearings and compare averages over different test sets, with no "
                      "equivalence margin. So this does not show that bearing overlap is harmless, or measure its "
                      "effect on the archived split.")


def markdown(res: dict) -> str:
    o, lobo, rep = res["original_split"], res["leave_one_bearing_out"], res["repeated_summary"]
    design, boot, sens = res["lobo_design"], res["lobo_bootstrap"], res["lobo_validation_sensitivity"]
    n = rep["a8_gmm"]["recording"]["n_splits"] if rep else 0
    lines = [
        "# Evaluation protocols",
        "",
        f"Generated by `python scripts/evaluate_protocols.py` (bearing_audio {res['version']}, "
        f"commit `{res['provenance']['git_commit'][:12]}`). Recording-level results use mean aggregation over the "
        "2-second windows of a recording. Fault F1 treats *fault* as the positive class. Section 1 reproduces the "
        "thesis; sections 2–3 are a follow-up evaluation of the two selected thesis pipelines, not part of the "
        "original thesis experiments.",
        "",
        "## 1. Archived thesis split (9 test recordings: 3 healthy, 6 fault)",
        "",
        "| Pipeline | Recording confusion | Fault F1 | Balanced accuracy | MCC | Window confusion (81) | Misclassified |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for k, v in o.items():
        r = v["recording"]
        lines.append(f"| {NAMES[k]} | {cm(r)} | {r['fault_f1']:.3f} | {r['balanced_accuracy']:.3f} | {r['mcc']:.2f} "
                     f"| {cm(v['window'])} | {', '.join(v['misclassified']) or '–'} |")
    lines += [
        "",
        "## 2. Leave-one-bearing-out (60 recordings, 20 bearings)",
        "",
        "**Design (protocol v1).** Each of the 20 folds holds out one bearing (its 3 recordings) for testing. "
        "Three other bearings are the validation set, and all models train on the remaining 16 bearings. Only the "
        "GMM uses the validation set, to set its decision threshold; the MLP and the baseline ignore it, so all three "
        "models train on the same data. The validation bearings are the next three IDs after the test bearing. "
        f"Bearing IDs are grouped by fault type, so in {design['single_type_validation_folds']} of "
        f"{design['n_folds']} folds the validation faults all come from one type. "
        + ("Section 2c varies this choice. " if sens["n_selections"] else "")
        + "Every fold, threshold and held-out score is in `protocol_comparison.json`.",
        "",
        f"**Uncertainty.** The 95% intervals come from resampling the {boot['n_bearings']} bearings "
        f"({boot['n_resamples']:,} resamples of the fixed out-of-fold predictions). They leave out the variation "
        "from refitting the models and from the earlier choice of pipeline settings.",
        "",
    ] + lobo_rows(lobo, boot["balanced_accuracy_95"])
    lines += ["", "Paired differences in balanced accuracy (same resampled bearings for both pipelines), 95% interval:",
              ""]
    for pair, (lo, hi) in boot["paired_difference_95"].items():
        a, b = pair.split(" - ")
        lines.append(f"- {NAMES[a]} minus {NAMES[b]}: {lo:+.3f} to {hi:+.3f}")
    lines += ["", "Fault recall by fault type (inner race I, outer race O, ball B, cage C):", ""]
    for k, v in lobo.items():
        lines.append(f"- {NAMES[k]}: " + ", ".join(f"{c} {num(x)}" for c, x in v["recall_by_condition"].items()))
    nb = res["leave_one_bearing_out_without_ball_fault_bearings"]
    lines += [
        "",
        "### 2b. Same, without the ball-fault bearings 11–15 (45 recordings, 15 bearings)",
        "",
        "According to the dataset description, ball-fault data were recorded without load, while healthy, inner-race, "
        "outer-race and cage data were recorded at 400 N. Dropping bearings 11–15 removes that possible load cue.",
        "",
    ] + lobo_rows(nb)
    if sens["n_selections"]:
        lines += [
            "",
            f"### 2c. Sensitivity to the validation bearings ({sens['n_selections']} alternative selections)",
            "",
            "Each selection draws the three validation bearings of every fold at random from three different fault "
            "types. This changes the GMM threshold, and for all models which 16 bearings they train on.",
            "",
            "| Pipeline | Protocol v1 | Minimum | Median | Maximum |",
            "|---|---:|---:|---:|---:|",
        ]
        for k, v in sens["balanced_accuracy"].items():
            a = np.array(v)
            lines.append(f"| {NAMES[k]} | {lobo[k]['balanced_accuracy']:.3f} | {a.min():.3f} | {np.median(a):.3f} "
                         f"| {a.max():.3f} |")
    if rep:
        lines += [
            "",
            f"## 3. Repeated matched splits ({n} per protocol, thesis sizes 42/9/9)",
            "",
            "*Recording* splits shuffle recordings within each class, as the thesis did. *Bearing* splits keep all "
            "recordings of a physical bearing in one subset. Both use the same seeds, but they select different test "
            f"recordings: on average the two test sets share {res['repeated_design']['mean_shared_test_recordings']:.1f} "
            "of 9 recordings. Per-split results are in `repeated_splits.csv`.",
            "",
            "| Pipeline | Protocol | Balanced accuracy, mean ± SD | 10th–90th percentile | Fault F1, mean ± SD | Splits with F1 ≥ 0.9 |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for k, v in rep.items():
            for p in ("recording", "bearing"):
                s = v[p]
                lines.append(f"| {NAMES[k]} | {p} | {s['balanced_accuracy_mean']:.3f} ± {s['balanced_accuracy_sd']:.3f} "
                             f"| {s['balanced_accuracy_p10']:.2f}–{s['balanced_accuracy_p90']:.2f} "
                             f"| {s['fault_f1_mean']:.3f} ± {s['fault_f1_sd']:.3f} | {s['share_fault_f1_at_least_0.9']:.0%} |")
        lines += [
            "",
            "The SD is the spread between whole random splits, not a confidence interval. Difference between the "
            "protocols (recording minus bearing, mean balanced accuracy), and how the archived split ranks among the "
            "recording-level splits:",
            "",
        ]
        for k, v in rep.items():
            d = v["recording_minus_bearing_balanced_accuracy"]
            lines.append(f"- {NAMES[k]}: {'0.000' if abs(d) < 5e-4 else format(d, '+.3f')}; "
                         f"{v['recording']['share_splits_at_least_original_balanced_accuracy']:.0%} of random splits "
                         f"score at least the archived split's {o[k]['recording']['balanced_accuracy']:.3f}")
        diffs = {k: v["recording_minus_bearing_balanced_accuracy"] for k, v in rep.items()}
        lines += ["", protocol_difference_note(diffs, n)]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=200, help="matched splits per protocol (0 to skip)")
    ap.add_argument("--sensitivity", type=int, default=10, help="alternative validation selections (0 to skip)")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--cache", type=Path, default=ROOT / "artifacts" / "features.npz")
    args = ap.parse_args()

    t0 = time.time()
    recordings = list_recordings()
    res = {"version": __version__, "provenance": provenance(recordings)}
    table = FeatureTable(recordings, cache=args.cache)
    print(f"{len(table.rec)} windows from {len(table.recordings)} recordings ({time.time() - t0:.0f} s)", flush=True)

    res["original_split"] = original(table, load_original_split())
    for k, v in res["original_split"].items():
        print(f"original split  {NAMES[k]:26s} {cm(v['recording'])}  F1 {v['recording']['fault_f1']:.3f}"
              f"  balanced acc {v['recording']['balanced_accuracy']:.3f}", flush=True)

    folds = lobo_folds(table.recordings)
    family = fault_family(table.recordings)
    res["lobo_design"] = {
        "n_folds": len(folds), "train_bearings_per_fold": 16, "validation_bearings_per_fold": 3,
        "validation_choice": "next three bearing IDs after the test bearing (protocol v1)",
        "validation_used_by": ["a8_gmm"],
        "single_type_validation_folds": sum(len({family[b] for b in f["validation_bearings"]}) == 1 for f in folds),
    }
    res["leave_one_bearing_out"] = leave_one_bearing_out(table)
    res["lobo_bootstrap"] = bearing_bootstrap(res["leave_one_bearing_out"], table.recordings)
    res["leave_one_bearing_out_without_ball_fault_bearings"] = leave_one_bearing_out(table, BALL_FAULT_BEARINGS)
    for label, key in (("leave-one-bearing-out", "leave_one_bearing_out"),
                       ("  without bearings 11-15", "leave_one_bearing_out_without_ball_fault_bearings")):
        for k, v in res[key].items():
            print(f"{label:24s} {NAMES[k]:26s} {cm(v)}  F1 {v['fault_f1']:.3f}  balanced acc {v['balanced_accuracy']:.3f}",
                  flush=True)

    sens = {"method": "three validation bearings per fold, drawn from three different fault types",
            "seeds": list(range(args.sensitivity)), "n_selections": args.sensitivity,
            "balanced_accuracy": {k: [] for k in PIPELINES}}
    for seed in range(args.sensitivity):
        alt = leave_one_bearing_out(table, choose_validation=family_balanced_validation(table.recordings, seed))
        for k in PIPELINES:
            sens["balanced_accuracy"][k].append(alt[k]["balanced_accuracy"])
        print(f"  validation selection {seed + 1}/{args.sensitivity}", flush=True)
    res["lobo_validation_sensitivity"] = sens

    res["repeated_summary"], per_split = {}, []
    if args.repeats:
        shared = []
        for seed in range(args.repeats):
            by_recording, by_bearing = matched_splits(table, seed)
            shared.append(len(set(by_recording[2]) & set(by_bearing[2])))
        res["repeated_design"] = {"n_repeats": args.repeats, "mean_shared_test_recordings": float(np.mean(shared)),
                                  "identical_test_sets": int(sum(x == 9 for x in shared))}
        tick = lambda i, n: print(f"  repeated splits {i}/{n}", flush=True) if i % 25 == 0 or i == n else None
        scores = repeated(table, args.repeats, progress=tick, records=per_split)
        res["repeated_summary"] = summarise_repeated(scores, reference=res["original_split"])

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "protocol_comparison.json").write_text(json.dumps(clean(res), indent=1, allow_nan=False) + "\n")
    (args.out / "protocol_comparison.md").write_text(markdown(res))
    if per_split:
        with open(args.out / "repeated_splits.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_split[0]))
            w.writeheader()
            w.writerows({**r, "fault_f1": round(r["fault_f1"], 6), "balanced_accuracy": round(r["balanced_accuracy"], 6)}
                        for r in per_split)
    print(f"wrote {args.out / 'protocol_comparison.md'} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
