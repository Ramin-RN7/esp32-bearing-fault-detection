#!/usr/bin/env python3
"""
A47 — Export A8 GMM (after PCA4) to JSON for device scoring.

Writes:
  artifacts/exports/A8_GMM/gmm.json
  artifacts/exports/A8_GMM/preprocess.json
"""
import argparse, json, sys, glob
from pathlib import Path
import numpy as np, joblib

def find_file(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits: return hits[-1]
    return None

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=r"artifacts/gmm_A8.joblib",
                    help="Joblib dict/Pipeline with scaler(Z), pca(4), gmm")
    ap.add_argument("--out", default=r"artifacts/exports/A8_GMM")
    ap.add_argument("--thr", type=float, default=None, help="Optional decision threshold override")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    jl = Path(args.bundle)
    if not jl.exists():
        alt = find_file([r"artifacts/runs/A8-GMM_*/bundle.joblib",
                         r"artifacts/runs/**/A8*/*.joblib",
                         r"artifacts/gmm_A8.joblib"])
        if not alt:
            print(f"[ERR] bundle not found: {jl}")
            sys.exit(2)
        jl = Path(alt)

    obj = joblib.load(jl)
    scaler=pca=gmm=None; thr=None
    if isinstance(obj, dict):
        scaler = obj.get('scaler') or obj.get('scale')
        pca    = obj.get('pca')
        gmm    = obj.get('gmm') or obj.get('model') or obj.get('clf')
        thr    = obj.get('thr', None)
    else:
        # pipeline or raw estimator
        try:
            from sklearn.pipeline import Pipeline
            if isinstance(obj, Pipeline):
                for n,s in obj.steps:
                    n=n.lower()
                    if 'pca' in n: pca=s
                    if ('scaler' in n) or ('standard' in n) or ('minmax' in n): scaler=s
                gmm = obj.steps[-1][1]
            else:
                gmm = obj
        except Exception:
            gmm = obj

    # write preprocess
    pre = {"feature_set":"A8","scale":scaler_params(scaler),"pca":pca_params(pca),"classes":["0","1"]}
    with open(out_dir/"preprocess.json","w",encoding="utf-8") as f:
        json.dump(pre, f, indent=2)
    print(f"[OK] wrote preprocess: {out_dir/'preprocess.json'}")

    from sklearn.mixture import GaussianMixture
    if not isinstance(gmm, GaussianMixture):
        print("[ERR] final estimator is not GaussianMixture")
        sys.exit(3)

    covtype = gmm.covariance_type
    if covtype not in ("diag","spherical","full","tied"):
        print(f"[WARN] unusual covariance_type={covtype}; proceeding")

    export = {
        "covariance_type": covtype,
        "weights": gmm.weights_.astype(float).tolist(),
        "means": gmm.means_.astype(float).tolist(),
        "covariances": gmm.covariances_.astype(float).tolist(),
        "thr": float(args.thr) if args.thr is not None else (float(thr) if thr is not None else 0.0),
    }
    with open(out_dir/"gmm.json","w",encoding="utf-8") as f:
        json.dump(export, f, indent=2)
    print(f"[OK] wrote gmm: {out_dir/'gmm.json'}")
    print("[DONE] A47 export complete.")

if __name__=="__main__":
    main()
