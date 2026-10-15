from pathlib import Path
import json, shutil, datetime
import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "manifests"
ART = ROOT / "artifacts"

# params used so far
PARAMS = {
  "sample_rate": 16000,
  "window_seconds": 2.0,
  "overlap": 0.5,
  "nfft": 4096,
  "features": ["rms","zcr","centroid_hz","bandwidth_hz","flatness",
               "kurtosis","crest","bp_0_1k","bp_1_3k","bp_3_8k","hf_ratio"]
}

def load_split(name):
    df = pd.read_csv(MAN / f"features_{name}.csv")
    y = (df["class"] == "anomaly").astype(int).values
    X = df[PARAMS["features"]].values.astype(np.float32)
    return X, y, df

# Recompute test metrics from saved calibrated models (if present)
metrics = {}
try:
    scaler = load(ART / "scaler.joblib")
    gmm = load(ART / "gmm_model_cal.joblib")
    mlp = load(ART / "mlp_model_cal.joblib")
    with open(ART / "gmm_calibration.json") as f:
        gcal = json.load(f)
    th = gcal["threshold"]

    Xtr, ytr, Dtr = load_split("train")
    Xva, yva, Dva = load_split("val")
    Xte, yte, Dte = load_split("test")

    Xte_s = scaler.transform(Xte)
    g_scores = gmm.score_samples(Xte_s)
    yp_gmm = (g_scores < th).astype(int)
    yp_mlp = mlp.predict(Xte_s)

    def m(y, yp):
        return dict(
            acc=float(accuracy_score(y, yp)),
            prec=float(precision_score(y, yp, zero_division=0)),
            rec=float(recall_score(y, yp)),
            f1=float(f1_score(y, yp)),
            cm=confusion_matrix(y, yp).tolist()
        )
    metrics["GMM_calibrated"] = m(yte, yp_gmm)
    metrics["MLP_tiny_scaled"] = m(yte, yp_mlp)
except Exception as e:
    metrics["note"] = f"Could not recompute metrics: {e}"

# Make bundle dir with timestamp
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
B = ROOT / f"report_bundle_{ts}"
(B / "manifests").mkdir(parents=True, exist_ok=True)
(B / "artifacts").mkdir(parents=True, exist_ok=True)

# Copy key files
for name in ["segments.csv","segments_split.csv","split_train.csv","split_val.csv","split_test.csv",
             "features_train.csv","features_val.csv","features_test.csv"]:
    src = MAN / name
    if src.exists():
        shutil.copy2(src, B / "manifests" / name)

for p in ART.glob("*"):
    shutil.copy2(p, B / "artifacts" / p.name)

# Write params & metrics
with open(B / "params.json","w") as f: json.dump(PARAMS, f, indent=2)
with open(B / "metrics.json","w") as f: json.dump(metrics, f, indent=2)

# Minimal README
readme = f"""# Bearing ML Report Bundle

Created: {ts}

## Parameters
- Sample rate: {PARAMS['sample_rate']} Hz
- Window: {PARAMS['window_seconds']} s, overlap {PARAMS['overlap']*100:.0f}%
- NFFT: {PARAMS['nfft']}
- Features: {', '.join(PARAMS['features'])}

## Test metrics
{json.dumps(metrics, indent=2)}

## Files
- manifests/: segments, splits, features
- artifacts/: models (.joblib), calibration JSON, confusion matrices
- params.json, metrics.json
"""
with open(B / "README_report.md","w",encoding="utf-8") as f: f.write(readme)

print("Bundle:", B)
