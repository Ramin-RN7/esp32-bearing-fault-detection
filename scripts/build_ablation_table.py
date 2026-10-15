#!/usr/bin/env python3
"""Build results/ablation_runs.csv from the original run log and per-run parameters.

It replaces 06_results_evaluation/ablation_summary.csv, whose configuration columns contain errors, for example
A8, A16, A17 and A31 being marked "no PCA". Every row comes from records/logs/variants.csv, the experiment log
written by the ablation scripts, enriched with records/runs/<run>/params.json. It also adds what the log does not
state:

* evaluation level: 81 test windows, or 9 test recordings for the file-aggregation runs
* confusion counts recovered from the logged accuracy, precision and recall
* balanced accuracy and MCC
* whether the run is equivalent to predicting "fault" for everything

All metrics are test-set metrics, as logged.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "records" / "logs" / "variants.csv"
RUNS = ROOT / "records" / "runs"
OUT = ROOT / "results" / "ablation_runs.csv"
RECORDING_LEVEL_PREFIXES = ("A40-", "A41-", "B1-", "B2-")


def level(row):
    if row["model"].lower().endswith("-file") or "-FILE" in row["id"] or row["id"].startswith(RECORDING_LEVEL_PREFIXES):
        return "recording", 6, 3
    return "window", 54, 27


def counts(acc, prec, rec, pos, neg):
    tp = round(rec * pos)
    tn = round(acc * (pos + neg)) - tp
    fp, fn = neg - tn, pos - tp
    ok = (abs((tp / (tp + fp) if tp + fp else 0.0) - prec) < 1e-6 and 0 <= tn <= neg)
    return tp, fp, tn, fn, ok


def params_summary(run_dir: Path) -> dict:
    p = run_dir / "params.json"
    if not p.exists():
        return {}
    j = json.loads(p.read_text())
    pre = j.get("preprocess", "")
    return {
        "sample_rate": j.get("sr", j.get("sample_rate", "")),
        "preprocess": " + ".join(pre) if isinstance(pre, list) else pre,
        "features": j.get("features", ""),
        "scaler": j.get("scaler", ""),
        "pca_dims": j.get("pca_dims", ""),
        "decision": j.get("mode", j.get("rule", j.get("threshold", ""))),
    }


def main():
    rows = list(csv.DictReader(open(LOG, encoding="utf-8")))
    last = {r["id"]: i for i, r in enumerate(rows)}
    out = []
    for i, r in enumerate(rows):
        lvl, pos, neg = level(r)
        acc, prec, rec, f1 = (float(r[k]) for k in ("acc", "precision", "recall", "f1"))
        tp, fp, tn, fn, ok = counts(acc, prec, rec, pos, neg)
        bal = (tp / pos + tn / neg) / 2
        den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = (tp * tn - fp * fn) / den if den else 0.0
        run_dir = RUNS / Path(r["run_dir"].replace("\\", "/")).name
        out.append({
            "id": r["id"], "timestamp": r["timestamp"], "latest_for_id": i == last[r["id"]],
            "description": r["change"], "model": r["model"], "level": lvl, "test_units": pos + neg,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4), "fault_f1": round(f1, 4),
            "balanced_accuracy": round(bal, 4), "mcc": round(mcc, 4),
            "equals_always_fault": tp == pos and tn == 0, "counts_consistent": ok,
            **params_summary(run_dir), "run_record": f"records/runs/{run_dir.name}" if run_dir.exists() else "",
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in out for k in row))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    bad = [o["id"] for o in out if not o["counts_consistent"]]
    trivial = sorted({o["id"] for o in out if o["equals_always_fault"]})
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out)} rows, {len(last)} distinct IDs")
    print(f"rows equivalent to always predicting 'fault': {len(trivial)} IDs -> {', '.join(trivial)}")
    if bad:
        print(f"rows whose logged metrics are not internally consistent: {bad}", file=sys.stderr)


if __name__ == "__main__":
    main()
