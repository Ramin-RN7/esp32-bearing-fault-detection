#!/usr/bin/env python3
r"""
A45 quick stub — export A44 ExtraTrees as RF placeholder so pipeline can proceed.
Writes:
  artifacts/exports/A45_RF/model_rf.h
  artifacts/exports/A45_RF/preprocess.json
"""
import json, sys
from pathlib import Path
import joblib, numpy as np

MODEL_IN  = Path(r"artifacts\et_A44.joblib")
OUT_DIR   = Path(r"artifacts\exports\A45_RF")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_IN.exists():
    print(f"[ERR] missing {MODEL_IN}"); sys.exit(2)

print(f"[..] loading: {MODEL_IN}")
bundle = joblib.load(MODEL_IN)

# --- try to grab scaler/pca/classes from the bundle
def find(obj, pred):
    from sklearn.pipeline import Pipeline
    if pred(obj): return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = find(v, pred);
            if r is not None: return r
    if isinstance(obj, (list,tuple)):
        for v in obj:
            r = find(v, pred);
            if r is not None: return r
    try:
        if isinstance(obj, Pipeline):
            for _, s in obj.steps:
                r = find(s, pred);
                if r is not None: return r
    except Exception:
        pass
    return None

scaler = find(bundle, lambda x: x.__class__.__name__ in ("MinMaxScaler","StandardScaler"))
pca    = find(bundle, lambda x: x.__class__.__name__ == "PCA")
clf    = find(bundle, lambda x: hasattr(x, "predict") and hasattr(x, "fit"))

def scaler_params(s):
    if s is None: return None
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    if isinstance(s, MinMaxScaler):
        return {"type":"minmax",
                "feature_range": list(getattr(s,"feature_range",(0.0,1.0))),
                "data_min": s.data_min_.astype(float).tolist(),
                "data_max": s.data_max_.astype(float).tolist(),
                "scale": s.scale_.astype(float).tolist(),
                "min": s.min_.astype(float).tolist()}
    if isinstance(s, StandardScaler):
        return {"type":"z",
                "mean": s.mean_.astype(float).tolist(),
                "scale": s.scale_.astype(float).tolist(),
                "var": s.var_.astype(float).tolist(),
                "with_mean": bool(getattr(s,"with_mean",True)),
                "with_std": bool(getattr(s,"with_std",True))}
    return None

def pca_params(p):
    if p is None: return None
    return {
        "n_components": int(getattr(p,"n_components_", getattr(p,"n_components",0)) or 0),
        "components": np.array(p.components_, dtype=float).tolist(),
        "mean": np.array(getattr(p,"mean_", np.zeros(len(p.components_[0]))), dtype=float).tolist(),
        "explained_variance_ratio": np.array(getattr(p,"explained_variance_ratio_", []), dtype=float).tolist(),
        "whiten": bool(getattr(p,"whiten", False)),
    }

classes = [str(c) for c in getattr(clf, "classes_", ["0","1"])]
pre = {"feature_set":"A16", "scale": scaler_params(scaler), "pca": pca_params(pca), "classes": classes}
(OUT_DIR/"preprocess.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
print(f"[OK] wrote preprocess: {OUT_DIR/'preprocess.json'}")

try:
    import emlearn
except Exception:
    print("[ERR] emlearn not installed in this interpreter.")
    print(r'Fix: & "C:\Users\ramin\AppData\Local\Programs\Python\Python310\python.exe" -m pip install -U emlearn')
    sys.exit(3)

print("[WARN] using A44 ExtraTrees as A45 placeholder (not a true RF).")
cmodel = emlearn.convert(clf, method="inline", dtype="float32", return_type="classifier")
hdr = OUT_DIR/"model_rf.h"
cmodel.save(name="A45_RF", file=str(hdr))
print(f"[OK] header written: {hdr}")
print("[DONE] A45 stub export complete.")
