#!/usr/bin/env python3
"""Freeze the A5 reference that an ESP32 port is checked against (2026 follow-up).

    python scripts/freeze_reference.py                 # writes reference/a5/ (about a minute)
    python scripts/freeze_reference.py --skip-float32  # without the float32 emulation over all 540 windows

Writes to reference/a5/:

* models/lobo_bearing_NN.json: the 20 leave-one-bearing-out models of A5 MLP, trained exactly as in
  results/ablation_lobo.md (16 training bearings each, seed 42), and models/all_bearings.json, trained on all 20.
* window_features.csv: the 25 features of all 540 windows, with P(fault) from the window's fold model and from
  the all-bearings model.
* decisions.csv: the 60 recording decisions of both.
* golden/<recording>.json: every stage of every window of four recordings, from PCM samples to the decision.
* contract.json: the constants, tolerances and file hashes of docs/DECISION_CONTRACT.md, for a port to load.
* float32_check.json: how far the float32 emulation (src/bearing_audio/port32.py) lands from the reference, as a
  fraction of each tolerance.

The features come from src/bearing_audio/reference.py (no librosa). The script stops if they differ from the
archived feature code in any bit, or if the leave-one-bearing-out decisions or scores differ from the
published run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.exceptions import ConvergenceWarning                       # noqa: E402

from bearing_audio import port32 as P                                    # noqa: E402
from bearing_audio import reference as R                                 # noqa: E402
from bearing_audio.ablation import Fitted, ablation_features, configs    # noqa: E402
from bearing_audio.data import list_recordings                           # noqa: E402
from bearing_audio.evaluation import lobo_folds                          # noqa: E402
from bearing_audio.features import FEATURE_NAMES, file_sha256            # noqa: E402

OUT = ROOT / "reference" / "a5"
GOLDEN = ("H_10_0", "B_11_2", "H_13_0", "C_19_2", "H_2_0")   # see docs/DECISION_CONTRACT.md, section 2


def dump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-float32", action="store_true", help="skip the float32 emulation over all 540 windows")
    args = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    t0 = time.time()
    recs = list_recordings()
    by_name = {r.name: r for r in recs}

    # 1. Features of every window, from the reference
    normalized = {r.name: R.normalize(R.read_pcm(r.path))[0] for r in recs}
    feats, gaps, names = [], [], []
    for r in recs:
        for w in R.windows(normalized[r.name]):
            notch = R.front_end(R.window_signal(w))["notch"]
            feats.append(R.features25(notch))
            gaps.append(R.lag_gap(notch.astype(np.float32)))
            names.append(r.name)
    X, names = np.vstack(feats), np.array(names)
    labels = np.array([by_name[n].label for n in names])
    print(f"features: {X.shape[0]} windows ({time.time() - t0:.0f} s)")

    cfg = next(c for c in configs() if c.id == "A5-MLP")
    archived = ablation_features(recs, [cfg], cache_dir=ROOT / "artifacts" / "ablation_features")["A5-MLP"]
    if not np.array_equal(X, archived):
        sys.exit("The reference features differ from the archived A5 feature code.")
    print("reference features equal the archived A5 feature code in every bit")

    # 2. Models: the 20 leave-one-bearing-out folds and one model on all bearings
    meta = {"scikit_learn": version("scikit-learn"), "seed": cfg.seed,
            "training": "ablation.Fitted with the A5-MLP configuration (as results/ablation_lobo.md)"}
    folds = lobo_folds(by_name)
    models, fold_of = {}, {}
    for f in folds:
        tr = np.isin(names, f["train"])
        fit = Fitted(cfg, X[tr], labels[tr], X[:0], labels[:0])     # the MLP doesn't use validation data
        key = f"lobo_bearing_{f['test_bearing']:02d}"
        models[key] = (fit, R.export_model(fit, model=key, test_bearing=f["test_bearing"],
                                           trained_on=sorted(f["train"]), **meta))
        for n in f["test"]:
            fold_of[n] = key
    fit_all = Fitted(cfg, X, labels, X[:0], labels[:0])
    models["all_bearings"] = (fit_all, R.export_model(fit_all, model="all_bearings", test_bearing=None,
                                                      trained_on=sorted(by_name), **meta))
    for key, (_, m) in models.items():
        dump(m, OUT / "models" / f"{key}.json")

    # 3. Window probabilities and recording decisions; check them against the published run
    p_fold = np.array([R.model_stages(X[i], models[fold_of[n]][1])["probability"][0] for i, n in enumerate(names)])
    p_all = np.array([R.model_stages(X[i], models["all_bearings"][1])["probability"][0] for i in range(len(X))])
    sk_fold = np.zeros(len(X))
    for r in recs:                                       # in one batch per recording, as run_split() does
        m = names == r.name
        sk_fold[m] = models[fold_of[r.name]][0].window_scores(X[m])
    published = {p["recording"]: p for p in json.loads((ROOT / "results" / "ablation_lobo.json").read_text())
                 ["lobo"]["A5-MLP"]["predictions"]}
    rows = []
    for r in recs:
        m = names == r.name
        s_fold, d_fold = R.recording_decision(p_fold[m])
        s_all, d_all = R.recording_decision(p_all[m])
        pub = published[r.name]
        if d_fold != pub["pred"] or float(np.mean(sk_fold[m].astype(np.float32))) != pub["score"]:
            sys.exit(f"{r.name}: leave-one-bearing-out result differs from results/ablation_lobo.json")
        rows.append({"recording": r.name, "bearing": r.bearing, "truth": r.label, "fold_model": fold_of[r.name],
                     "lobo_score": repr(s_fold), "lobo_decision": d_fold, "lobo_correct": int(d_fold == r.label),
                     "lobo_margin": f"{abs(s_fold - R.THRESHOLD):.4f}",
                     "all_bearings_score": repr(s_all), "all_bearings_decision": d_all})
    print("leave-one-bearing-out decisions and scores equal the published run; "
          f"JSON forward pass within {np.abs(p_fold - sk_fold).max():.1e} of scikit-learn")
    with open(OUT / "decisions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "window_features.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["recording", "window", *FEATURE_NAMES, "lag_gap", "p_lobo", "p_all_bearings"])
        k = 0
        for r in recs:
            for i in range(int(np.sum(names == r.name))):
                w.writerow([r.name, i + 1, *[f"{v:.9g}" for v in X[k]], f"{gaps[k]:.6g}", repr(float(p_fold[k])),
                            repr(float(p_all[k]))])
                k += 1

    # 4. Golden outputs
    spread = np.asarray(models["all_bearings"][1]["standardize"]["scale"])
    for name in GOLDEN:
        rec, key = by_name[name], fold_of[name]
        tr = R.trace(rec.path, models[key][1])
        tr_all = R.trace(rec.path, models["all_bearings"][1])
        dump({"recording": name, "bearing": rec.bearing, "truth": rec.label, "model": f"models/{key}.json",
              "wav": str(rec.path.relative_to(ROOT)), "wav_sha256": file_sha256(rec.path),
              "gain": float(tr["gain"]),
              "normalized_sha256": hashlib.sha256(tr["normalized"].astype("<i2").tobytes()).hexdigest(),
              "score": tr["score"], "decision": tr["decision"], "margin": abs(tr["score"] - R.THRESHOLD),
              "windows": [R.golden_window(w, i) for i, w in enumerate(tr["windows"])],
              "all_bearings": {"model": "models/all_bearings.json",
                               "probabilities": [float(w["probability"][0]) for w in tr_all["windows"]],
                               "score": tr_all["score"], "decision": tr_all["decision"]}},
             OUT / "golden" / f"{name}.json")

    # 5. The contract's constants, for a port to load
    contract = {
        "pipeline": "A5-MLP", "document": "docs/DECISION_CONTRACT.md",
        "input": {"format": "WAV, 16-bit signed PCM, little-endian, mono", "sample_rate": R.SAMPLE_RATE},
        "normalize": {"peak_target": R.PEAK_TARGET, "peak_target_float32": float(np.float32(R.PEAK_TARGET)),
                      "steps": ["y = pcm / 32768 (float32)", "gain = float32(min(1, peak_target / max|y|))",
                                "n = int16(trunc(clip(y * gain, -1, 1) * 32767)), float32 arithmetic"]},
        "windows": {"length": R.WINDOW, "hop": R.HOP,
                    "steps": ["x = n / 32767 (float32)",
                              "w = int16(trunc(clip(x[start:start+32000], -1, 1) * 32767)), float32 arithmetic",
                              "signal = w / 32768 (float32)"]},
        "highpass": {"type": "Butterworth high-pass", "order": R.HPF_ORDER, "cutoff_hz": R.HPF_HZ,
                     "b": R.HPF_B.tolist(), "a": R.HPF_A.tolist(), "sos": R.HPF_SOS.tolist(),
                     "initial_state": "zero at the start of every window"},
        "preemphasis": {"coefficient": R.PREEMPHASIS, "first_sample": "unchanged", "output": "float32"},
        "notch": {"f0_hz": R.NOTCH_HZ, "q": R.NOTCH_Q, "b": R.NOTCH_B.tolist(), "a": R.NOTCH_A.tolist(),
                  "zi": R.NOTCH_ZI.tolist(), "pad": R.NOTCH_PAD, "padding": "odd",
                  "steps": ["ext = odd extension by pad samples at each end",
                            "forward pass from state zi * ext[0]",
                            "backward pass over the reversed result from state zi * (its first sample)",
                            "reverse, drop pad samples at each end"]},
        "features": {"names": list(FEATURE_NAMES), "input": "notch output as float32", "dft_points": R.WINDOW,
                     "bands_hz": [list(b) for b in R.BANDS_HZ], "zero_crossing_threshold": R.ZC_THRESHOLD,
                     "stft": {"n_fft": R.N_FFT, "hop": R.STFT_HOP, "window": "periodic Hann", "frames": 63,
                              "centred": True, "padding": "zeros"},
                     "rolloff_percent": R.ROLL_PERCENT, "flatness_amin": R.FLATNESS_AMIN, "max_lag": R.MAX_LAG,
                     "spread": spread.tolist()},
        "decision": {"window_score": "P(fault) from the MLP", "recording_score": "mean of the 9 window scores",
                     "threshold": R.THRESHOLD, "fault_when": "recording_score >= threshold",
                     "labels": {"0": "healthy", "1": "fault"}},
        "tolerances": {"pcm": "identical", "signal_fraction_of_peak": R.SIGNAL_TOLERANCE,
                       "feature_fraction_of_spread": R.FEATURE_TOLERANCE, "lag_near_tie": R.LAG_NEAR_TIE,
                       "model_from_golden_features": R.MODEL_TOLERANCE, "probability": R.PROBABILITY_TOLERANCE,
                       "decisions": "identical for all 60 recordings"},
        "probes": list(R.PROBES),
        "models": sorted(f"models/{k}.json" for k in models),
        "golden": [f"golden/{n}.json" for n in GOLDEN],
        "audio_sha256": {r.name: file_sha256(r.path) for r in recs},
        "versions": {"python": platform.python_version(),
                     **{k: version(k) for k in ("numpy", "scipy", "scikit-learn", "soundfile")}},
    }
    dump(contract, OUT / "contract.json")

    # 6. The float32 emulation against the contract, over all 540 windows
    if not args.skip_float32:
        t1 = time.time()
        worst: dict[str, float] = {}
        pw, k = [], 0
        for r in recs:
            port_normalized, _ = P.normalize(R.read_pcm(r.path))
            if not np.array_equal(port_normalized, normalized[r.name]):
                sys.exit(f"{r.name}: float32 normalisation differs")
            model = models[fold_of[r.name]][1]
            for i, w in enumerate(R.windows(port_normalized)):
                st = {"pcm": w, **P.front_end(R.window_signal(w))}
                st["features"] = P.features25(st["notch"])
                st["probability"] = P.model_stages(st["features"], model)["probability"]
                ref = {"pcm": w, **R.front_end(R.window_signal(w))}
                ref["features"] = X[k]
                golden = R.golden_window({**ref, **R.model_stages(X[k], model)}, i)
                for c, v in R.check_window(st, golden, spread).items():
                    worst[c] = max(worst.get(c, 0.0), v)
                pw.append(float(st["probability"][0]))
                k += 1
        pw = np.array(pw)
        rec_diff = max(abs(np.mean(pw[names == r.name]) - np.mean(p_fold[names == r.name])) for r in recs)
        same = all((np.mean(pw[names == r.name]) >= R.THRESHOLD) == (np.mean(p_fold[names == r.name]) >= R.THRESHOLD)
                   for r in recs)
        worst["recording_score"] = rec_diff / R.PROBABILITY_TOLERANCE["recording"]
        dump({"what": "largest difference of the float32 emulation (port32.py) from the reference over all 540 "
                      "windows, as a fraction of each tolerance; 1.0 is at the limit",
              "windows": int(k), "decisions_identical": bool(same),
              "largest_window_probability_difference": float(np.abs(pw - p_fold).max()),
              "largest_recording_score_difference": float(rec_diff),
              "fraction_of_tolerance": {c: round(v, 4) for c, v in worst.items()}},
             OUT / "float32_check.json")
        print(f"float32 emulation: worst check at {max(worst.values()):.2f} of its tolerance, decisions identical: "
              f"{same} ({time.time() - t1:.0f} s)")
        if max(worst.values()) > 1 or not same:
            sys.exit("The float32 emulation does not meet the contract.")
    print(f"wrote {OUT.relative_to(ROOT)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
