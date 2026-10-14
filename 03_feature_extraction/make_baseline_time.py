from pathlib import Path, PurePath
import json, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from joblib import dump

ROOT = Path(__file__).resolve().parent
MAN = ROOT / "manifests"
ART = ROOT / "artifacts"; ART.mkdir(exist_ok=True)

FEATS = ["rms","zcr","kurtosis","crest"]  # time-domain only (easy on device)

def load(name):
    df = pd.read_csv(MAN / f"features_{name}.csv")
    y = (df["class"]=="anomaly").astype(int).values
    X = df[FEATS].values.astype(np.float32)
    return X,y

Xtr,ytr = load("train"); Xva,yva = load("val"); Xte,yte = load("test")

sc = StandardScaler().fit(Xtr)
Xtr_s, Xva_s, Xte_s = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)

clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=0).fit(Xtr_s, ytr)

def metr(y, yp):
    return dict(acc=accuracy_score(y,yp), prec=precision_score(y,yp,zero_division=0),
                rec=recall_score(y,yp), f1=f1_score(y,yp),
                cm=confusion_matrix(y,yp).tolist())

m_te = metr(yte, clf.predict(Xte_s))
print("== Baseline(time-domain) test ==")
print(f"acc={m_te['acc']:.3f} P={m_te['prec']:.3f} R={m_te['rec']:.3f} F1={m_te['f1']:.3f}")
print("CM test:", m_te["cm"])

model = {
  "type": "logreg",
  "features": FEATS,
  "scaler_mean": sc.mean_.tolist(),
  "scaler_scale": sc.scale_.tolist(),
  "weights": clf.coef_[0].tolist(),
  "bias": float(clf.intercept_[0]),
  "threshold": 0.5
}
out = ART / "baseline_logreg_time.json"
with open(out,"w") as f: json.dump(model, f, indent=2)
print("Saved:", out)
