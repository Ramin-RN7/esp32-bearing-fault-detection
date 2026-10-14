#!/usr/bin/env python3
# scans artifacts for a joblib that contains sklearn.neural_network.MLPClassifier
import glob, joblib, sys
from pathlib import Path

def walk(o):
    yield o
    try:
        from sklearn.pipeline import Pipeline
        if isinstance(o, dict):
            for v in o.values(): yield from walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o: yield from walk(v)
        elif isinstance(o, Pipeline):
            for _, s in o.steps: yield from walk(s)
    except Exception:
        pass

hits = []
candidates = (glob.glob(r"artifacts\mlp_*.joblib", recursive=True) +
              glob.glob(r"artifacts\runs\**\*.joblib", recursive=True))
for p in sorted(set(candidates)):
    try:
        obj = joblib.load(p)
    except Exception:
        continue
    for x in walk(obj):
        try:
            from sklearn.neural_network import MLPClassifier
            if isinstance(x, MLPClassifier):
                layers = [w.shape for w in x.coefs_]
                hits.append((p, layers, len(getattr(x, "classes_", [])) or 2))
                break
        except Exception:
            pass

if not hits:
    print("[ERR] no MLPClassifier found in artifacts. Pass --pipeline to a known A16-MLP run, or retrain MLP.")
    sys.exit(1)

print("[OK] Found MLP joblib(s):")
for p, layers, ncls in hits[-5:]:
    print(f"  {p}  layers={layers}  classes={ncls}")
print("\nUse the most recent path with the exporter via --pipeline.")
