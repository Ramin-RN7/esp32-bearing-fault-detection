#!/usr/bin/env python3
r"""
A46 — minimal MLP exporter → C header + preprocess.json
Use:  python a46_export_mlp_MIN.py --pipeline "artifacts\\runs\\A9-MLP_20250813_234816\\mlp_A9.joblib"
Writes:
  artifacts\exports\A16_MLP\preprocess.json
  artifacts\exports\A16_MLP\model_mlp.h
"""
import argparse, json, sys, glob
from pathlib import Path
import joblib, numpy as np

# --- tiny utils ---
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

def first(root, types):
    for x in walk(root):
        if isinstance(x, types): return x
    return None

def sp_scaler(s):
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

def sp_pca(p):
    if p is None: return None
    return {
        "n_components": int(getattr(p,"n_components_", getattr(p,"n_components",0)) or 0),
        "components": np.array(p.components_, dtype=float).tolist(),
        "mean": np.array(getattr(p,"mean_", np.zeros(len(p.components_[0]))), dtype=float).tolist(),
        "explained_variance_ratio": np.array(getattr(p,"explained_variance_ratio_", []), dtype=float).tolist(),
        "whiten": bool(getattr(p,"whiten", False)),
    }

# --- main ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline", required=True, help="Path to joblib containing an sklearn MLPClassifier")
    ap.add_argument("--out", default=r"artifacts\exports\A16_MLP")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    jl = Path(args.pipeline)
    if not jl.exists():
        print(f"[ERR] pipeline not found: {jl}"); sys.exit(2)

    print(f"[..] loading: {jl}")
    root = joblib.load(jl)

    # find parts
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.neural_network import MLPClassifier

    scaler = first(root, (MinMaxScaler, StandardScaler))
    pca = first(root, (PCA,))
    mlp = first(root, (MLPClassifier,))
    if mlp is None:
        print("[ERR] no MLPClassifier found in this joblib"); sys.exit(3)

    # preprocess.json
    classes = [str(c) for c in getattr(mlp, "classes_", ["0","1"])]
    pre = {"feature_set":"A16", "scale": sp_scaler(scaler), "pca": sp_pca(pca), "classes": classes}
    (out/"preprocess.json").write_text(json.dumps(pre, indent=2), encoding="utf-8")
    print(f"[OK] wrote preprocess: {out/'preprocess.json'}")

    # header from weights
    coefs = [w.astype(np.float32) for w in mlp.coefs_]
    biases = [b.astype(np.float32) for b in mlp.intercepts_]
    hidden_act = getattr(mlp, "activation", "relu")
    out_act = "softmax" if len(mlp.classes_)>2 else "logistic"

    hp = out/"model_mlp.h"
    with open(hp, "w", encoding="utf-8") as h:
        h.write("/* Auto-generated MLP header (A16_MLP) */\n#pragma once\n#include <math.h>\n#include <stdint.h>\n\n")
        # weights & biases
        for i,(W,B) in enumerate(zip(coefs, biases)):
            h.write(f"static const int L{i}_IN={W.shape[0]};\nstatic const int L{i}_OUT={W.shape[1]};\n")
            h.write(f"static const float L{i}_W[{W.size}]={{"+",".join(f"{x:.8e}" for x in W.ravel(order='F'))+"};\n")
            h.write(f"static const float L{i}_B[{B.size}]={{"+",".join(f"{x:.8e}" for x in B)+"};\n")
        h.write("\nstatic inline float act_relu(float x){return x>0?x:0;}\n")
        h.write("static inline float act_tanhf(float x){return tanhf(x);} \n")
        h.write("static inline float act_logistic(float x){return 1.0f/(1.0f+expf(-x));}\n")
        h.write("static void dense_forward(const float *W,const float *B,int IN,int OUT,const float *x,float *y){\n")
        h.write("  for(int o=0;o<OUT;o++){float s=B[o];const float *col=&W[o*IN];for(int i=0;i<IN;i++){s+=col[i]*x[i];}y[o]=s;}}\n")
        h.write("\n// returns argmax class index\nstatic int A16_MLP_predict(const float *x_in,float *tmp0,float *tmp1){\n")
        h.write("  const float *x=x_in; float *a=tmp0; float *b=tmp1; int IN,OUT;\n")
        for i,(W,B) in enumerate(zip(coefs, biases)):
            h.write(f"  IN=L{i}_IN; OUT=L{i}_OUT; dense_forward(L{i}_W,L{i}_B,IN,OUT,x,a);\n")
            if i < len(coefs)-1:
                h.write("  for(int j=0;j<OUT;j++){")
                h.write("a[j]=act_relu(a[j]);" if hidden_act=="relu" else
                        "a[j]=act_tanhf(a[j]);" if hidden_act=="tanh" else
                        "a[j]=act_logistic(a[j]);" if hidden_act=="logistic" else
                        "/*identity*/")
                h.write("}\n  const float *sw=x; x=a; a=(float*)sw;\n")
            else:
                if out_act=="softmax":
                    h.write("  float m=a[0]; for(int j=1;j<OUT;j++){ if(a[j]>m) m=a[j]; }\n")
                    h.write("  float sum=0.0f; for(int j=0;j<OUT;j++){ a[j]=expf(a[j]-m); sum+=a[j]; }\n")
                    h.write("  int arg=0; float best=a[0]/sum; for(int j=1;j<OUT;j++){ float p=a[j]/sum; if(p>best){best=p; arg=j;} }\n  return arg;\n")
                else:
                    h.write("  float p=act_logistic(a[0]); return p>=0.5f?1:0;\n")
        h.write("}\n")
    print(f"[OK] wrote header: {hp}")
    print("[DONE] A46 export complete.")

if __name__ == "__main__":
    main()
