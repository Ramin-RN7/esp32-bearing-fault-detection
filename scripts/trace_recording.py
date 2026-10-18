#!/usr/bin/env python3
"""Every A5 stage of one recording, as the frozen reference computes it (2026 follow-up).

    python scripts/trace_recording.py H_13_0                       # all 9 windows, with the recording's fold model
    python scripts/trace_recording.py H_13_0 --model all_bearings  # with the model trained on all 20 bearings
    python scripts/trace_recording.py H_13_0 --float32             # plus the float32 emulation, checked against
                                                                   # the contract's tolerances
    python scripts/trace_recording.py H_13_0 --window 3 --save w3.npz    # every stage of window 3 as full arrays
    python scripts/trace_recording.py O_8_1 --json O_8_1.json      # a golden-format record for any recording

Needs reference/a5/ (scripts/freeze_reference.py). The stages and tolerances are described in
docs/DECISION_CONTRACT.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bearing_audio import port32 as P              # noqa: E402
from bearing_audio import reference as R           # noqa: E402
from bearing_audio.data import list_recordings     # noqa: E402

REF = ROOT / "reference" / "a5"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="e.g. H_13_0")
    ap.add_argument("--model", default="lobo", help="lobo (the recording's fold model, default) or all_bearings")
    ap.add_argument("--float32", action="store_true", help="also run the float32 emulation and check it")
    ap.add_argument("--window", type=int, help="window number (1-9) for --save")
    ap.add_argument("--save", type=Path, help="write every stage of --window as arrays to this .npz file")
    ap.add_argument("--json", type=Path, help="write a golden-format record of the recording to this file")
    a = ap.parse_args()
    recs = {r.name: r for r in list_recordings()}
    if a.recording not in recs:
        sys.exit(f"unknown recording {a.recording}; e.g. {', '.join(list(recs)[:3])}")
    rec = recs[a.recording]
    with open(REF / "decisions.csv", newline="") as f:
        fold = {row["recording"]: row["fold_model"] for row in csv.DictReader(f)}
    key = fold[rec.name] if a.model == "lobo" else a.model
    model = json.loads((REF / "models" / f"{key}.json").read_text())
    spread = np.asarray(json.loads((REF / "contract.json").read_text())["features"]["spread"])

    tr = R.trace(rec.path, model)
    truth = "fault" if rec.label else "healthy"
    print(f"{rec.name} (bearing {rec.bearing}, {truth}) with models/{key}.json; gain {float(tr['gain']):.9g}")
    head = (f"{'win':>3}  {'PCM SHA-256':12}  {'notch peak':>10}  {'notch rms':>9}  {'ZCR':>7}  {'lag':>3}  "
            f"{'P(fault)':>8}")
    print(head + ("  float32: worst check (fraction of tolerance)" if a.float32 else ""))
    golden = []
    for i, w in enumerate(tr["windows"]):
        g = R.golden_window(w, i)
        golden.append(g)
        line = (f"{i + 1:>3}  {g['pcm']['sha256'][:12]}  {g['notch']['peak']:10.6f}  {g['notch']['rms']:9.6f}  "
                f"{w['features'][9]:7.5f}  {int(w['features'][24]):>3}  {g['probability']:8.5f}")
        if a.float32:
            st = {"pcm": w["pcm"], **P.front_end(R.window_signal(w["pcm"]))}
            st["features"] = P.features25(st["notch"])
            st["probability"] = P.model_stages(st["features"], model)["probability"]
            checks = R.check_window(st, g, spread)
            worst = max(checks, key=checks.get)
            line += f"  {worst} {checks[worst]:.3f}"
        print(line)
    print(f"score {tr['score']:.6f} -> {'fault' if tr['decision'] else 'healthy'} "
          f"({'correct' if tr['decision'] == rec.label else 'wrong'}; margin {abs(tr['score'] - R.THRESHOLD):.4f})")

    if a.save:
        if not a.window:
            sys.exit("--save needs --window")
        w = tr["windows"][a.window - 1]
        np.savez(a.save, **{k: np.asarray(v) for k, v in w.items()})
        print(f"wrote {a.save} ({', '.join(w)})")
    if a.json:
        a.json.write_text(json.dumps({"recording": rec.name, "bearing": rec.bearing, "truth": rec.label,
                                      "model": f"models/{key}.json", "gain": float(tr["gain"]),
                                      "score": tr["score"], "decision": tr["decision"], "windows": golden}, indent=1))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
