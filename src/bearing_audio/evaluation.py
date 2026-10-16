"""Evaluation protocols.

* ``original``: the archived thesis split (42/9/9 recordings), which reproduces the thesis numbers exactly.
* ``leave_one_bearing_out`` (protocol v1): every physical bearing is held out once (all three of its recordings),
  giving 60 out-of-fold recording decisions. In each fold three other bearings form the validation set and all
  models train on the remaining 16 bearings. Only the GMM uses the validation set, to set its threshold; the MLP and
  the baseline ignore it, so all three models train on the same data. The validation bearings are the next three
  IDs after the test bearing (``adjacent_validation``). Bearing IDs are grouped by fault type, so the validation
  faults often come from a single type; ``family_balanced_validation`` draws them from three different types
  instead, as a sensitivity check.
* ``repeated``: matched random splits with the thesis sizes (train 42 / val 9 / test 9 recordings, test =
  3 healthy + 6 fault). ``recording`` splits shuffle recordings within each class like make_splits.py;
  ``bearing`` splits assign whole bearings (14/3/3), so no bearing crosses subsets. Both protocols use
  the same seeds.
* ``bearing_bootstrap``: percentile intervals from resampling bearings, on fixed out-of-fold predictions.
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
        rec_true, rec_pred, rec_score, win_true, win_pred = [], [], [], [], []
        for name in test:
            m = table.rec == name
            rec_true.append(table.recordings[name].label)
            rec_score.append(model.recording_score(X[m]))
            rec_pred.append(model.predict_recording(X[m]))
            win_true += table.label[m].tolist()
            win_pred += model.predict_windows(X[m]).tolist()
        out[key] = {
            "test": [str(n) for n in test], "true": rec_true, "pred": rec_pred, "score": rec_score,
            "threshold": float(model.threshold),
            "recording": metrics(rec_true, rec_pred), "window": metrics(win_true, win_pred),
            "misclassified": [str(n) for n, t, p in zip(test, rec_true, rec_pred) if t != p],
        }
    return out


def original(table: FeatureTable, split: dict[str, str]) -> dict:
    pick = lambda s: [n for n in sorted(table.recordings) if split[n] == s]
    return evaluate_split(table, pick("train"), pick("val"), pick("test"))


def adjacent_validation(bearings, test_bearing, index):
    """Protocol v1: the next three bearing IDs after the test bearing, wrapping around."""
    rest = [b for b in bearings if b != test_bearing]
    return {rest[(index + j) % len(rest)] for j in range(3)}


def fault_family(recordings: dict) -> dict[int, str]:
    """Bearing ID -> fault type of its fault recordings (I, O, B or C)."""
    return {r.bearing: r.condition for r in recordings.values() if r.label == 1}


def family_balanced_validation(recordings: dict, seed: int):
    """Sensitivity check: three validation bearings drawn at random from three different fault types."""
    family = fault_family(recordings)
    rng = np.random.default_rng(seed)

    def choose(bearings, test_bearing, index):
        rest = [b for b in bearings if b != test_bearing]
        types = rng.choice(sorted({family[b] for b in rest}), size=3, replace=False)
        return {int(rng.choice([b for b in rest if family[b] == t])) for t in types}
    return choose


def lobo_folds(recordings: dict, exclude_bearings=(), choose_validation=adjacent_validation) -> list[dict]:
    """The leave-one-bearing-out folds: test, validation and training recordings, all grouped by bearing."""
    names = sorted(n for n in recordings if recordings[n].bearing not in set(exclude_bearings))
    bear = {n: recordings[n].bearing for n in names}
    bearings = sorted(set(bear.values()))
    folds = []
    for i, b in enumerate(bearings):
        val_b = set(choose_validation(bearings, b, i))
        folds.append({
            "test_bearing": b, "validation_bearings": sorted(val_b),
            "train": [n for n in names if bear[n] != b and bear[n] not in val_b],
            "val": [n for n in names if bear[n] in val_b],
            "test": [n for n in names if bear[n] == b],
        })
    return folds


def leave_one_bearing_out(table: FeatureTable, exclude_bearings=(), choose_validation=adjacent_validation) -> dict:
    """Out-of-fold recording decisions with each bearing held out once.

    ``exclude_bearings`` drops all recordings of those bearings first. For example, bearings 11-15 are the
    ball-fault bearings, whose fault recordings were made without load. Each pipeline's result lists every fold
    (test and validation bearings, decision threshold) and every held-out recording with its continuous score.
    """
    pooled = {k: {"folds": [], "predictions": []} for k in PIPELINES}
    for fold in lobo_folds(table.recordings, exclude_bearings, choose_validation):
        res = evaluate_split(table, fold["train"], fold["val"], fold["test"])
        for k, r in res.items():
            pooled[k]["folds"].append({"test_bearing": fold["test_bearing"],
                                       "validation_bearings": fold["validation_bearings"],
                                       "n_train_recordings": len(fold["train"]), "threshold": r["threshold"]})
            pooled[k]["predictions"] += [
                {"recording": n, "test_bearing": fold["test_bearing"], "true": t, "pred": p, "score": sc}
                for n, t, p, sc in zip(r["test"], r["true"], r["pred"], r["score"])]
    out = {}
    for k, v in pooled.items():
        preds = v["predictions"]
        true, pred = [p["true"] for p in preds], [p["pred"] for p in preds]
        recs = [table.recordings[p["recording"]] for p in preds]

        def recall_where(cond):
            idx = [i for i, r in enumerate(recs) if r.label == 1 and cond(r)]
            return float(np.mean([pred[i] for i in idx])) if idx else float("nan")
        out[k] = {
            **metrics(true, pred),
            "recall_developing_fault": recall_where(lambda r: r.state == 1),
            "recall_faulty": recall_where(lambda r: r.state == 2),
            "recall_by_condition": {c: recall_where(lambda r, c=c: r.condition == c) for c in "IOBC"},
            "misclassified": [p["recording"] for p in preds if p["true"] != p["pred"]],
            "folds": v["folds"],
            "predictions": preds,
        }
    return out


BOOTSTRAP_PAIRS = (("a16_mlp", "baseline_rms_kurtosis"), ("a16_mlp", "a8_gmm"), ("a8_gmm", "baseline_rms_kurtosis"))


def bearing_bootstrap(lobo: dict, recordings: dict, n_resamples: int = 10000, seed: int = 42) -> dict:
    """95% percentile intervals for balanced accuracy, resampling the bearings with replacement.

    The out-of-fold predictions stay fixed, so the intervals reflect which bearings happen to be in the dataset.
    They leave out the variation from refitting the models and from the earlier choice of pipeline settings.
    Every resample draws the same bearings for all pipelines, which gives the paired differences.
    """
    keys = list(lobo)
    bearings = sorted({recordings[p["recording"]].bearing for p in lobo[keys[0]]["predictions"]})
    index = {b: i for i, b in enumerate(bearings)}
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(bearings), size=(n_resamples, len(bearings)))
    counts = np.zeros((n_resamples, len(bearings)))
    np.add.at(counts, (np.repeat(np.arange(n_resamples), len(bearings)), draws.ravel()), 1)
    resampled = {}
    for k in keys:
        tally = np.zeros((4, len(bearings)))            # healthy n, healthy correct, fault n, fault correct
        for p in lobo[k]["predictions"]:
            i, row = index[recordings[p["recording"]].bearing], 2 * int(p["true"])
            tally[row, i] += 1
            tally[row + 1, i] += int(p["pred"] == p["true"])
        specificity, recall = counts @ tally[1] / (counts @ tally[0]), counts @ tally[3] / (counts @ tally[2])
        resampled[k] = (specificity + recall) / 2
    interval = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]
    return {
        "n_resamples": n_resamples, "seed": seed, "n_bearings": len(bearings),
        "balanced_accuracy_95": {k: interval(v) for k, v in resampled.items()},
        "paired_difference_95": {f"{a} - {b}": interval(resampled[a] - resampled[b])
                                 for a, b in BOOTSTRAP_PAIRS if a in resampled and b in resampled},
    }


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


def repeated(table: FeatureTable, n_repeats: int = 200, progress=None, records: list | None = None) -> dict:
    """(fault F1, balanced accuracy) per split. If ``records`` is a list, one row per split and pipeline is added."""
    scores = {p: {k: [] for k in PIPELINES} for p in ("recording", "bearing")}
    for seed in range(n_repeats):
        rec_split, bear_split = matched_splits(table, seed)
        for protocol, (train, val, test) in (("recording", rec_split), ("bearing", bear_split)):
            res = evaluate_split(table, train, val, test)
            for k in PIPELINES:
                m = res[k]["recording"]
                scores[protocol][k].append((m["fault_f1"], m["balanced_accuracy"]))
                if records is not None:
                    records.append({"seed": seed, "protocol": protocol, "pipeline": k,
                                    "test_recordings": " ".join(test), "fault_f1": m["fault_f1"],
                                    "balanced_accuracy": m["balanced_accuracy"]})
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
