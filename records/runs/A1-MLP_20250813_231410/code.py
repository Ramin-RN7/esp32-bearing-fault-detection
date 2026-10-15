# a1_hpf.py — A1: High-Pass Filter (~100 Hz) with robust paths + auto logging (same style as A0)
from pathlib import Path
from datetime import datetime
import numpy as np, joblib, json
import matplotlib.pyplot as plt
import librosa, soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import skew, kurtosis
from scipy.signal import butter, lfilter

# ---------- robust paths ----------
HERE = Path(__file__).resolve().parent
# Prefer project-local data; fallback to parent/Desktop layout if needed
DATA_CAND = [
    HERE / "data" / "segments_split",
    HERE.parent / "data" / "segments_split",
]
DATA = next((p for p in DATA_CAND if p.exists()), DATA_CAND[0])

# Keep logging where you already have variants.csv if it exists; otherwise use project artifacts
ART_CAND = [
    HERE.parent / "artifacts",   # e.g., Desktop\artifacts (your A0)
    HERE / "artifacts",          # project\artifacts
]
ART = next((p for p in ART_CAND if p.exists()), ART_CAND[0])
RUNS = ART / "runs"
ART.mkdir(parents=True, exist_ok=True)
RUNS.mkdir(parents=True, exist_ok=True)

print(f"[PATH] DATA = {DATA}")
print(f"[PATH] ART  = {ART}")

# ---------- constants ----------
SR = 16000
WIN_S = 2.0
OVERLAP = 0.5
RNG = 42

# ---------- audio utils ----------
def load_wav(fp):
    y, sr = sf.read(fp, dtype='float32', always_2d=False)
    if y.ndim > 1: y = y[:,0]
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y

def highpass(y, sr, cutoff=100.0, order=5):
    nyq = 0.5 * sr
    b, a = butter(order, cutoff/nyq, btype='high', analog=False)
    return lfilter(b, a, y)

def bandpower(y, sr, fmin, fmax):
    S = np.abs(np.fft.rfft(y))**2
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    idx = (freqs>=fmin)&(freqs<fmax)
    return float(S[idx].sum()/(S.sum()+1e-12))

def spectral_slope(y):
    S = np.abs(np.fft.rfft(y))
    x = np.arange(len(S))
    xm, ym = x.mean(), S.mean()
    num = ((x-xm)*(S-ym)).sum()
    den = ((x-xm)**2).sum() + 1e-12
    return float(num/den)

def autocorr_feats(y, max_lag=200):
    y = y - y.mean()
    ac = np.correlate(y, y, mode='full')[len(y)-1:len(y)+max_lag]
    if len(ac) < 2: return 0.0, 0
    ac[0] = 0.0
    k = int(np.argmax(ac))
    return float(ac[k]/(np.sum(y**2)+1e-12)), int(k)

# ---------- 25 features ----------
def features25(y, sr):
    if len(y) < int(SR*WIN_S):
        y = np.pad(y, (0, int(SR*WIN_S)-len(y)))
    x = y.astype(np.float32)
    # time (10)
    rms = float(np.sqrt(np.mean(x**2)+1e-12))
    feat = [
        float(np.mean(x)), float(np.std(x)), rms,
        float(np.max(np.abs(x))), float(np.ptp(x)),
        float(skew(x)), float(kurtosis(x,fisher=True)),
        float(np.mean(np.abs(np.diff(x)))),
        float(np.mean(np.square(np.diff(x)))),
        float(np.count_nonzero(librosa.zero_crossings(x)))/len(x)
    ]
    # freq (13)
    bands = [(0,250),(250,500),(500,1000),(1000,2000),
             (2000,3000),(3000,4000),(4000,6000),(6000,8000)]
    for f1,f2 in bands:
        feat.append(bandpower(x, sr, f1, f2))
    S = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), 1/sr)
    centroid = float((freqs*(S+1e-12)).sum()/(S.sum()+1e-12))
    bw = float(np.sqrt(((freqs-centroid)**2 * (S+1e-12)).sum()/(S.sum()+1e-12)))
    roll = float(librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85).mean())
    flat = float(librosa.feature.spectral_flatness(y=x).mean())
    slope = spectral_slope(x)
    feat += [centroid, bw, roll, flat, slope]
    # autocorr (2)
    ac_val, ac_lag = autocorr_feats(x)
    feat += [ac_val, float(ac_lag)]
    return np.asarray(feat, dtype=np.float32)

def load_split(split):
    base = DATA / split
    normals  = sorted((base/"normal").glob("*.wav"))
    anomalies= sorted((base/"anomaly").glob("*.wav"))
    if not normals and not anomalies:
        raise FileNotFoundError(f"No WAVs under {base}/(normal|anomaly)")
    X, y, paths = [], [], []
    for cls_list, label in [(normals,0),(anomalies,1)]:
        for fp in cls_list:
            sig = load_wav(fp)
            sig = highpass(sig, SR, cutoff=100.0)  # HPF step
            X.append(features25(sig, SR)); y.append(label); paths.append(str(fp))
    return np.vstack(X), np.array(y), paths

# ---------- metrics & saving ----------
def metrics_of(y_true, y_pred):
    return dict(
        acc=accuracy_score(y_true,y_pred),
        precision=precision_score(y_true,y_pred, zero_division=0),
        recall=recall_score(y_true,y_pred, zero_division=0),
        f1=f1_score(y_true,y_pred, zero_division=0),
        cm=confusion_matrix(y_true,y_pred)
    )

def save_cm(cm, name):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(name); plt.xlabel('Pred'); plt.ylabel('True')
    for (i,j),v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha='center', va='center')
    out = ART/f"cm_{name}.png"
    plt.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out

def log_all(run_id, change, model, metrics, params, cm_path, model_path, code_text):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = RUNS / f"{run_id}_{stamp}"
    rd.mkdir(parents=True, exist_ok=True)
    with open(rd/"params.json","w",encoding="utf-8") as f: json.dump(params, f, indent=2, ensure_ascii=False)
    mfull = dict(metrics); mfull["timestamp"]=stamp
    with open(rd/"metrics.json","w",encoding="utf-8") as f: json.dump(mfull, f, indent=2, ensure_ascii=False)
    with open(rd/"code.py","w",encoding="utf-8") as f: f.write(code_text)
    row = {
        "timestamp": stamp, "id": run_id, "change": change, "model": model,
        "acc": metrics.get("acc"), "precision": metrics.get("precision"),
        "recall": metrics.get("recall"), "f1": metrics.get("f1"),
        "run_dir": str(rd), "code_path": str(rd/"code.py"),
        "cm_path": str(cm_path), "params_path": str(rd/"params.json"),
        "metrics_path": str(rd/"metrics.json"), "model_path": str(model_path),
        "notes": ""
    }
    import csv
    csv_path = ART/"variants.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print(f"[OK] saved: {rd}\n[OK] appended: {csv_path}")

# ---------- 1) data ----------
X_train, y_train, _ = load_split("train")
X_val,   y_val,   _ = load_split("val")
X_test,  y_test,  _ = load_split("test")

# ---------- 2) scale + PCA ----------
scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train); Xv = scaler.transform(X_val); Xte = scaler.transform(X_test)
pca = PCA(n_components=4, random_state=RNG).fit(Xtr)
Ztr = pca.transform(Xtr); Zv = pca.transform(Xv); Zte = pca.transform(Xte)

# ---------- 3) GMM (OCC diag) ----------
HEALTHY = (y_train==0)
gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=RNG)
gmm.fit(Ztr[HEALTHY])

val_scores = gmm.score_samples(Zv)  # higher = more healthy
ths = np.percentile(val_scores, np.linspace(0.5,99.5,200))
best_f1, best_thr = -1.0, None
for t in ths:
    y_pred = (val_scores < t).astype(int)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    if f1 > best_f1: best_f1, best_thr = f1, t

test_scores = gmm.score_samples(Zte)
y_pred_gmm = (test_scores < best_thr).astype(int)
def metrics_of(y_true, y_pred):
    return dict(
        acc=accuracy_score(y_true,y_pred),
        precision=precision_score(y_true,y_pred, zero_division=0),
        recall=recall_score(y_true,y_pred, zero_division=0),
        f1=f1_score(y_true,y_pred, zero_division=0),
        cm=confusion_matrix(y_true,y_pred)
    )
m_gmm = metrics_of(y_test, y_pred_gmm)
cm_gmm_path = save_cm(m_gmm["cm"], "A1_GMM")
joblib.dump({"scaler":scaler,"pca":pca,"gmm":gmm,"thr":best_thr}, ART/"gmm_A1.joblib")

# ---------- 4) MLP tiny ----------
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam',
                    max_iter=500, random_state=RNG)
mlp.fit(Ztr, y_train)
y_pred_mlp = mlp.predict(Zte)
m_mlp = metrics_of(y_test, y_pred_mlp)
cm_mlp_path = save_cm(m_mlp["cm"], "A1_MLP")
joblib.dump({"scaler":scaler,"pca":pca,"mlp":mlp}, ART/"mlp_A1.joblib")

# ---------- 5) log ----------
params = dict(
    scaler="zscore", pca_dims=4, outlier_pct=0.05,
    threshold="val-opt", sr=SR, win_s=WIN_S, overlap=OVERLAP,
    preprocess=["hpf_100Hz"], rng=RNG
)
code_txt = open(__file__, 'r', encoding='utf-8').read()

log_all("A1-GMM", "HPF ~100 Hz", "GMM",
        {k:m_gmm[k] for k in ["acc","precision","recall","f1"]},
        {**params, "model_hparams": dict(n_components=3, cov_type="diag", thr=float(best_thr))},
        cm_gmm_path, ART/"gmm_A1.joblib", code_txt)

log_all("A1-MLP", "HPF ~100 Hz", "MLP",
        {k:m_mlp[k] for k in ["acc","precision","recall","f1"]},
        {**params, "model_hparams": dict(hidden=(10,), act="relu", solver="adam", max_iter=500)},
        cm_mlp_path, ART/"mlp_A1.joblib", code_txt)

print("\n=== A1 results ===")
print("GMM:", {k: round(m_gmm[k],3) for k in ["acc","precision","recall","f1"]}, "thr=", round(float(best_thr),3))
print("MLP:", {k: round(m_mlp[k],3) for k in ["acc","precision","recall","f1"]})
print(f"Saved to: {ART}  (CSV: {ART/'variants.csv'})")
