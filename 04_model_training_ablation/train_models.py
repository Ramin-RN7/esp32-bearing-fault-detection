from pathlib import Path
import csv, json, os
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "manifests"
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

FEAT = ["rms","zcr","centroid_hz","bandwidth_hz","flatness","kurtosis","crest",
        "bp_0_1k","bp_1_3k","bp_3_8k","hf_ratio"]

def load_csv(p):
    rows = []
    with open(p, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    X = np.array([[float(row[k]) for k in FEAT] for row in rows], dtype=np.float32)
    y = np.array([0 if row["class"]=="normal" else 1 for row in rows], dtype=np.int32)
    files = [row["filepath"] for row in rows]
    return X, y, files

def cm_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return cm, acc, pr, rc, f1

def save_cm(cm, labels, path, title):
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.xticks([0,1], labels)
    plt.yticks([0,1], labels)
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# --- load data ---
Xtr, ytr, ftr = load_csv(MAN/"features_train.csv")
Xva, yva, fva = load_csv(MAN/"features_val.csv")
Xte, yte, fte = load_csv(MAN/"features_test.csv")

# ---------------- GMM (healthy-only) ----------------
mask_norm = (ytr==0)
sc_gmm = StandardScaler().fit(Xtr[mask_norm])
XtrN = sc_gmm.transform(Xtr[mask_norm])
pca = PCA(n_components=min(8, XtrN.shape[1]), random_state=0).fit(XtrN)
XtrN_p = pca.transform(XtrN)

gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=0).fit(XtrN_p)

def gmm_scores(X):
    Xp = pca.transform(sc_gmm.transform(X))
    return gmm.score_samples(Xp)  # log-likelihood; higher is more "normal"

# pick threshold on VAL to maximize F1 (anomaly = score < threshold)
s_val = gmm_scores(Xva)
ths = np.linspace(np.min(s_val), np.max(s_val), 200)
best = {"th":None, "f1":-1, "cm":None, "acc":0, "pr":0, "rc":0}
for th in ths:
    pred = (s_val < th).astype(int)
    cm, acc, pr, rc, f1 = cm_metrics(yva, pred)
    if f1 > best["f1"]:
        best = {"th":float(th), "f1":float(f1), "cm":cm, "acc":float(acc), "pr":float(pr), "rc":float(rc)}
th_gmm = best["th"]

# evaluate on TEST with selected threshold
s_test = gmm_scores(Xte)
yhat_gmm = (s_test < th_gmm).astype(int)
cm_gmm_te, acc_gmm, pr_gmm, rc_gmm, f1_gmm = cm_metrics(yte, yhat_gmm)

# save GMM artifacts (joblib + JSON for portable params)
joblib.dump(sc_gmm, ART/"gmm_scaler.joblib")
joblib.dump(pca, ART/"gmm_pca.joblib")
joblib.dump(gmm, ART/"gmm_model.joblib")
with open(ART/"gmm_portable.json","w") as f:
    json.dump({
        "scaler_mean": sc_gmm.mean_.tolist(),
        "scaler_scale": sc_gmm.scale_.tolist(),
        "pca_components": pca.components_.tolist(),
        "pca_mean": pca.mean_.tolist(),
        "gmm_weights": gmm.weights_.tolist(),
        "gmm_means": gmm.means_.tolist(),
        "gmm_covariances": gmm.covariances_.tolist(),
        "threshold": th_gmm
    }, f, indent=2)

save_cm(cm_gmm_te, ["normal","anomaly"], ART/"cm_gmm_test.png", "GMM Test CM")
np.savetxt(ART/"scores_gmm_test.csv", np.c_[s_test, yte], delimiter=",", header="score_loglike,y_true", comments="")

# ---------------- Tiny MLP (supervised) ----------------
sc_mlp = StandardScaler().fit(Xtr)
Xtr_s = sc_mlp.transform(Xtr)
Xva_s = sc_mlp.transform(Xva)
Xte_s = sc_mlp.transform(Xte)

mlp = MLPClassifier(hidden_layer_sizes=(16,), activation="relu",
                    alpha=1e-4, max_iter=500, random_state=0,
                    early_stopping=True, n_iter_no_change=10).fit(Xtr_s, ytr)

yhat_mlp = mlp.predict(Xte_s)
cm_mlp_te, acc_mlp, pr_mlp, rc_mlp, f1_mlp = cm_metrics(yte, yhat_mlp)

joblib.dump(sc_mlp, ART/"mlp_scaler.joblib")
joblib.dump(mlp, ART/"mlp_model.joblib")
save_cm(cm_mlp_te, ["normal","anomaly"], ART/"cm_mlp_test.png", "MLP Test CM")

# ---------------- Summary ----------------
def sz(p): 
    try: return os.path.getsize(p)
    except: return 0

summary = f"""
== Test metrics ==
GMM(one-class, healthy-only): acc={acc_gmm:.3f}  P={pr_gmm:.3f}  R={rc_gmm:.3f}  F1={f1_gmm:.3f}  th={th_gmm:.4f}
MLP(tiny):                   acc={acc_mlp:.3f}  P={pr_mlp:.3f}  R={rc_mlp:.3f}  F1={f1_mlp:.3f}

Artifacts in: {ART}
  GMM: cm={ART/'cm_gmm_test.png'}, model(joblib)={ART/'gmm_model.joblib'} ({sz(ART/'gmm_model.joblib')} B), params(json)={ART/'gmm_portable.json'}
  MLP: cm={ART/'cm_mlp_test.png'}, model={ART/'mlp_model.joblib'} ({sz(ART/'mlp_model.joblib')} B)
"""
print(summary)

with open(ART/"metrics_summary.txt","w") as f:
    f.write(summary)
