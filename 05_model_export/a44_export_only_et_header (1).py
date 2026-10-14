#!/usr/bin/env python3
r"""
A44 — Export ExtraTrees to emlearn header + preprocess.json (robust auto-find).
"""
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np

def walk(obj):
    yield obj
    from sklearn.pipeline import Pipeline
    if isinstance(obj, dict):
        for v in obj.values(): yield from walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj: yield from walk(v)
    else:
        try:
            if isinstance(obj, Pipeline):
                for _, s in obj.steps: yield from walk(s)
        except Exception:
            pass

def first(instance_iter, types):
    for x in instance_iter:
        if isinstance(x, types): return x
    return None

def extract_preprocess(root):
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.decomposition import PCA
    scaler = first(walk(root), (MinMaxScaler, StandardScaler))
    pca    = first(walk(root), (PCA,))
    def scaler_params(s):
        if s is None: return None
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
        out={}
        for k in ["mean_","scale_","var_","data_min_","data_max_","min_"]:
            if hasattr(s,k): out[k.strip("_")] = getattr(s,k).astype(float).tolist()
        out["type"]=out.get("type","custom")
        return out or None
    def pca_params(p):
        if p is None: return None
        return {
            "n_components": int(getattr(p,"n_components_", getattr(p,"n_components",0)) or 0),
            "components": np.array(p.components_, dtype=float).tolist(),
            "mean": np.array(getattr(p,"mean_", np.zeros(len(p.components_[0]))), dtype=float).tolist(),
            "explained_variance_ratio": np.array(getattr(p,"explained_variance_ratio_", []), dtype=float).tolist(),
            "whiten": bool(getattr(p,"whiten", False)),
        }
    return scaler_params(scaler), pca_params(pca)

def find_classifier(root):
    # prefer ExtraTrees/RandomForest
    try:
        from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
        clf = first(walk(root), (ExtraTreesClassifier, RandomForestClassifier))
        if clf: return clf
    except Exception: pass
    # otherwise any classifier-like with predict
    for x in walk(root):
        if hasattr(x, "predict") and hasattr(x, "fit"):
            return x
    return None

def classes_of(clf):
    if hasattr(clf, "classes_"):
        return [str(c) for c in clf.classes_]
    return ["0","1"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"artifacts\et_A44.joblib")
    ap.add_argument("--out",   default=r"artifacts\exports\A44_ET")
    ap.add_argument("--name",  default="A44_ET")
    ap.add_argument("--leaf_bits", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32","float16"])
    args = ap.parse_args()

    mp = Path(args.model)
    od = Path(args.out); od.mkdir(parents=True, exist_ok=True)
    if not mp.exists():
        print(f"[ERR] model not found: {mp}"); sys.exit(2)

    print(f"[..] loading: {mp}")
    root = joblib.load(mp)

    scaler, pca = extract_preprocess(root)
    clf = find_classifier(root)
    if clf is None:
        print("[ERR] could not find classifier in bundle"); sys.exit(3)

    pre = {"feature_set":"A16","scale": scaler, "pca": pca, "classes": classes_of(clf)}
    (od/"preprocess.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
    print(f"[OK] wrote preprocess: {od/'preprocess.json'}")

    try:
        import emlearn
    except Exception:
        print("[ERR] emlearn not installed in this interpreter.")
        print("Fix: & \"C:\\Users\\ramin\\AppData\\Local\\Programs\\Python\\Python310\\python.exe\" -m pip install -U emlearn")
        sys.exit(4)

    print(f"[..] converting with emlearn (dtype={args.dtype}, leaf_bits={args.leaf_bits})")
    cmodel = emlearn.convert(clf, method="inline", dtype=args.dtype, leaf_bits=args.leaf_bits, return_type="classifier")
    hdr = od/"model_et.h"
    cmodel.save(name=args.name, file=str(hdr))
    print(f"[OK] emlearn header written: {hdr}")
    print("[DONE] A44 export complete.")

if __name__ == "__main__":
    main()
