"""Evaluation protocols.

* ``original``: the archived thesis split (42/9/9 recordings), which reproduces the thesis numbers exactly.
* ``leave_one_bearing_out``: every physical bearing is held out once (all three of its recordings), giving
  60 out-of-fold recording decisions. Three other bearings form the validation set for the GMM threshold.
* ``repeated``: matched random splits with the thesis sizes (train 42 / val 9 / test 9 recordings, test =
  3 healthy + 6 fault). ``recording`` splits shuffle recordings within each class like make_splits.py;
  ``bearing`` splits assign whole bearings (14/3/3), so no bearing crosses subsets. Both protocols use
  the same seeds.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef

from .features import FeatureTable
from .pipelines import PIPELINES


def metrics(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "accuracy": float((tp + tn) / len(y_true)),
        "fault_f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 and len(set(y_true)) > 1 else 0.0,
        "fault_recall": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
    }


def evaluate_split(table: FeatureTable, train, val, test, pipelines=PIPELINES) -> dict:
    tr, va = table.mask(train), table.mask(val)
    out = {}
    for key, cls in pipelines.items():
        X = table.features(cls.features)
        model = cls().fit(X[tr], table.label[tr], X[va], table.label[va])
        rec_true, rec_pred, win_true, win_pred = [], [], [], []
        for name in test:
            m = table.rec == name
            rec_true.append(table.recordings[name].label)
            rec_pred.append(model.predict_recording(X[m]))
            win_true += table.label[m].tolist()
            win_pred += model.predict_windows(X[m]).tolist()
        out[key] = {
            "test": [str(n) for n in test], "true": rec_true, "pred": rec_pred,
            "recording": metrics(rec_true, rec_pred), "window": metrics(win_true, win_pred),
            "misclassified": [str(n) for n, t, p in zip(test, rec_true, rec_pred) if t != p],
        }
    return out


def original(table: FeatureTable, split: dict[str, str]) -> dict:
    pick = lambda s: [n for n in sorted(table.recordings) if split[n] == s]
    return evaluate_split(table, pick("train"), pick("val"), pick("test"))


def leave_one_bearing_out(table: FeatureTable, exclude_bearings=()) -> dict:
    """Out-of-fold recording decisions with each bearing held out once.

    ``exclude_bearings`` drops all recordings of those bearings first. For example, bearings 11-15 are the
    ball-fault bearings, whose fault recordings were made without load.
    """
    names = sorted(n for n in table.recordings if table.recordings[n].bearing not in set(exclude_bearings))
    bear = {n: table.recordings[n].bearing for n in names}
    bearings = sorted(set(bear.values()))
    pooled = {k: {"names": [], "true": [], "pred": []} for k in PIPELINES}
    for i, b in enumerate(bearings):
        rest = [x for x in bearings if x != b]
        val_b = {rest[(i + j) % len(rest)] for j in range(3)}
        test = [n for n in names if bear[n] == b]
        val = [n for n in names if bear[n] in val_b]
        train = [n for n in names if bear[n] != b and bear[n] not in val_b]
        res = evaluate_split(table, train, val, test)
        for k in PIPELINES:
            pooled[k]["names"] += res[k]["test"]
            pooled[k]["true"] += res[k]["true"]
            pooled[k]["pred"] += res[k]["pred"]
    out = {}
    for k, v in pooled.items():
        recs = [table.recordings[n] for n in v["names"]]
        def recall_where(cond):
            idx = [i for i, r in enumerate(recs) if r.label == 1 and cond(r)]
            return float(np.mean([v["pred"][i] for i in idx])) if idx else float("nan")
        out[k] = {
            **metrics(v["true"], v["pred"]),
            "recall_developing_fault": recall_where(lambda r: r.state == 1),
            "recall_faulty": recall_where(lambda r: r.state == 2),
            "recall_by_condition": {c: recall_where(lambda r, c=c: r.condition == c) for c in "IOBC"},
            "misclassified": [n for n, t, p in zip(v["names"], v["true"], v["pred"]) if t != p],
            "predictions": [{"recording": n, "true": t, "pred": p} for n, t, p in zip(v["names"], v["true"], v["pred"])],
        }
    return out


def matched_splits(table: FeatureTable, seed: int):
    """One recording-level and one bearing-level split with the thesis sizes, drawn from the same seed.

    Returns ((train, val, test) by recording, (train, val, test) by bearing).
    """
    names = sorted(table.recordings)
    healthy = [n for n in names if table.recordings[n].label == 0]
    faulty = [n for n in names if table.recordings[n].label == 1]
    bear = {n: table.recordings[n].bearing for n in names}
    rng = np.random.default_rng(seed)
    h, f = [str(x) for x in rng.permutation(healthy)], [str(x) for x in rng.permutation(faulty)]
    by_recording = (h[:14] + f[:28], h[14:17] + f[28:34], h[17:] + f[34:])
    b = [int(x) for x in rng.permutation(sorted(set(bear.values())))]
    test_b, val_b = set(b[:3]), set(b[3:6])
    by_bearing = ([n for n in names if bear[n] not in test_b | val_b],
                  [n for n in names if bear[n] in val_b], [n for n in names if bear[n] in test_b])
    return by_recording, by_bearing


def repeated(table: FeatureTable, n_repeats: int = 200, progress=None) -> dict:
    scores = {p: {k: [] for k in PIPELINES} for p in ("recording", "bearing")}
    for seed in range(n_repeats):
        rec_split, bear_split = matched_splits(table, seed)
        for protocol, (train, val, test) in (("recording", rec_split), ("bearing", bear_split)):
            res = evaluate_split(table, train, val, test)
            for k in PIPELINES:
                m = res[k]["recording"]
                scores[protocol][k].append((m["fault_f1"], m["balanced_accuracy"]))
        if progress:
            progress(seed + 1, n_repeats)
    return scores


def summarise_repeated(scores: dict, reference: dict | None = None) -> dict:
    """Mean, spread and percentiles per protocol; optionally where the archived split ranks."""
    out = {}
    for protocol, per_pipe in scores.items():
        for k, vals in per_pipe.items():
            a = np.array(vals)
            f1, ba = a[:, 0], a[:, 1]
            entry = {
                "n_splits": int(len(a)),
                "fault_f1_mean": float(f1.mean()), "fault_f1_sd": float(f1.std()),
                "balanced_accuracy_mean": float(ba.mean()), "balanced_accuracy_sd": float(ba.std()),
                "balanced_accuracy_p10": float(np.percentile(ba, 10)),
                "balanced_accuracy_p50": float(np.percentile(ba, 50)),
                "balanced_accuracy_p90": float(np.percentile(ba, 90)),
                "share_fault_f1_at_least_0.9": float((f1 >= 0.9).mean()),
            }
            if reference is not None and protocol == "recording":
                ref = reference[k]["recording"]["balanced_accuracy"]
                entry["share_splits_at_least_original_balanced_accuracy"] = float((ba >= ref - 1e-12).mean())
            out.setdefault(k, {})[protocol] = entry
    for k in out:
        r = np.array(scores["recording"][k])[:, 1]
        b = np.array(scores["bearing"][k])[:, 1]
        out[k]["recording_minus_bearing_balanced_accuracy"] = float((r - b).mean())
    return out
