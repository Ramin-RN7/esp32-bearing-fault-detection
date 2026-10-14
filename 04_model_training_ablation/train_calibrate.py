from pathlib import Path
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "manifests"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

FEATS = ["rms","zcr","centroid_hz","bandwidth_hz","flatness","kurtosis","crest",
         "bp_0_1k","bp_1_3k","bp_3_8k","hf_ratio"]

def load_split(name):
    df = pd.read_csv(MAN / f"features_{name}.csv")
    y = (df["class"] == "anomaly").astype(int).values
    X = df[FEATS].values.astype(np.float32)
    return X, y, df

Xtr, ytr, Dtr = load_split("train")
Xva, yva, Dva = load_split("val")
Xte, yte, Dte = load_split("test")

# ---- Scale features (fit on train only) ----
scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)
Xva_s = scaler.transform(Xva)
Xte_s = scaler.transform(Xte)

# ---- One-class GMM on healthy ----
Xtr_h = Xtr_s[ytr==0]
gmm = GaussianMixture(n_components=1, covariance_type="diag", random_state=0)
gmm.fit(Xtr_h)

# pick threshold on val to maximize F1
val_scores = gmm.score_samples(Xva_s)  # higher = more healthy-like
ths = np.percentile(val_scores, np.linspace(1,99,199))  # candidate thresholds
best = (-1, None)  # (F1, threshold)
for th in ths:
    ypred = (val_scores < th).astype(int)  # anomaly if below threshold
    f1 = f1_score(yva, ypred)
    if f1 > best[0]:
        best = (f1, th)
best_f1, best_th = best

# evaluate on test
te_scores = gmm.score_samples(Xte_s)
yp_gmm = (te_scores < best_th).astype(int)

# ---- Tiny MLP (balanced) ----
mlp = MLPClassifier(hidden_layer_sizes=(16,), alpha=1e-4, max_iter=1000,
                    early_stopping=True, random_state=0)
# class balance via sample weights
w = np.where(ytr==1, (ytr==0).sum()/(ytr==1).sum(), 1.0)
sw = np.where(ytr==1, w, 1.0)
mlp.fit(Xtr_s, ytr, sample_weight=sw)

yp_mlp = mlp.predict(Xte_s)

def metrics(y, yp):
    return dict(
        acc = accuracy_score(y, yp),
        prec = precision_score(y, yp, zero_division=0),
        rec = recall_score(y, yp),
        f1 = f1_score(y, yp)
    )

m_gmm = metrics(yte, yp_gmm)
m_mlp = metrics(yte, yp_mlp)

print("== Calibrated Test metrics ==")
print(f"GMM(one-class)+scaler: acc={m_gmm['acc']:.3f}  P={m_gmm['prec']:.3f}  R={m_gmm['rec']:.3f}  F1={m_gmm['f1']:.3f}  th={best_th:.4f}")
print(f"MLP(tiny)+scaler:      acc={m_mlp['acc']:.3f}  P={m_mlp['prec']:.3f}  R={m_mlp['rec']:.3f}  F1={m_mlp['f1']:.3f}")

# save confusion matrices
def save_cm(y, yp, name):
    cm = confusion_matrix(y, yp)  # [[TN,FP],[FN,TP]]
    fig = plt.figure(figsize=(3,3))
    plt.imshow(cm, interpolation="nearest")
    plt.title(name); plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i,j]), ha="center", va="center")
    plt.tight_layout()
    out = ART / f"cm_{name}.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out

cm_g = save_cm(yte, yp_gmm, "gmm_test")
cm_m = save_cm(yte, yp_mlp, "mlp_test")

# save artifacts
dump(scaler, ART / "scaler.joblib")
dump(gmm, ART / "gmm_model_cal.joblib")
dump(mlp, ART / "mlp_model_cal.joblib")

with open(ART / "gmm_calibration.json","w") as f:
    json.dump({
        "threshold": float(best_th),
        "features": FEATS,
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "gmm": {
            "weights": gmm.weights_.tolist(),
            "means": gmm.means_.tolist(),
            "covariances": gmm.covariances_.tolist(),
            "cov_type": gmm.covariance_type
        }
    }, f, indent=2)

print("Artifacts in:", ART)
