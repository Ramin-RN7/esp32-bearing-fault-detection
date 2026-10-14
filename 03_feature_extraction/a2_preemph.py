# a2_preemph.py — A2: Pre-emphasis (alpha=0.97). Saves ONLY to project ./artifacts
from pathlib import Path
from datetime import datetime
import numpy as np, joblib, json, shutil
import matplotlib.pyplot as plt
import librosa, soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import skew, kurtosis

# ----- fixed project paths -----
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "segments_split"
ART  = HERE / "artifacts"
RUNS = ART / "runs"
ART.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
print(f"[PATH] DATA = {DATA}"); print(f"[PATH] ART  = {ART}")

# ----- constants -----
SR=16000; WIN_S=2.0; RNG=42; ALPHA=0.97

# ----- audio / features -----
def load_wav(fp):
    y, sr = sf.read(fp, dtype='float32', always_2d=False)
    if y.ndim>1: y = y[:,0]
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y

def preemph(y, alpha=ALPHA):
    if len(y)<2: return y
    out = np.empty_like(y, dtype=np.float32)
    out[0]=y[0]; out[1:]=y[1:] - alpha*y[:-1]
    return out

def bandpower(y, sr, fmin, fmax):
    S = np.abs(np.fft.rfft(y))**2
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    idx = (freqs>=fmin)&(freqs<fmax)
    return float(S[idx].sum()/(S.sum()+1e-12))

def spectral_slope(y):
    S = np.abs(np.fft.rfft(y)); x = np.arange(len(S))
    xm, ym = x.mean(), S.mean()
    return float(((x-xm)*(S-ym)).sum()/(((x-xm)**2).sum()+1e-12))

def autocorr_feats(y, max_lag=200):
    y = y - y.mean()
    ac = np.correlate(y, y, mode='full')[len(y)-1:len(y)+max_lag]
    if len(ac)<2: return 0.0, 0
    ac[0]=0.0; k=int(np.argmax(ac))
    return float(ac[k]/(np.sum(y**2)+1e-12)), int(k)

def features25(y, sr):
    if len(y) < int(SR*WIN_S):
        y = np.pad(y, (0, int(SR*WIN_S)-len(y)))
    x = y.astype(np.float32)
    rms = float(np.sqrt(np.mean(x**2)+1e-12))
    feat = [
        float(np.mean(x)), float(np.std(x)), rms,
        float(np.max(np.abs(x))), float(np.ptp(x)),
        float(skew(x)), float(kurtosis(x,fisher=True)),
        float(np.mean(np.abs(np.diff(x)))),
        float(np.mean(np.square(np.diff(x)))),
        float(np.count_nonzero(librosa.zero_crossings(x)))/len(x)
    ]
    bands=[(0,250),(250,500),(500,1000),(1000,2000),(2000,3000),(3000,4000),(4000,6000),(6000,8000)]
    for f1,f2 in bands: feat.append(bandpower(x, sr, f1, f2))
    S = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(len(x), 1/sr)
    centroid = float((freqs*(S+1e-12)).sum()/(S.sum()+1e-12))
    bw = float(np.sqrt(((freqs-centroid)**2*(S+1e-12)).sum()/(S.sum()+1e-12)))
    roll = float(librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85).mean())
    flat = float(librosa.feature.spectral_flatness(y=x).mean())
    slope = spectral_slope(x)
    ac_val, ac_lag = autocorr_feats(x)
    feat += [centroid, bw, roll, flat, slope, ac_val, float(ac_lag)]
    return np.asarray(feat, dtype=np.float32)

def load_split(split):
    base = DATA / split
    normals  = sorted((base/"normal").glob("*.wav"))
    anomalies= sorted((base/"anomaly").glob("*.wav"))
    if not normals and not anomalies:
        raise FileNotFoundError(f"No WAVs under {base}/(normal|anomaly)")
    X,y,paths = [],[],[]
    for files,label in [(normals,0),(anomalies,1)]:
        for fp in files:
            sig = preemph(load_wav(fp), ALPHA)
            X.append(features25(sig, SR)); y.append(label); paths.append(str(fp))
    return np.vstack(X), np.array(y), paths

# ----- metrics / logging -----
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
    plt.imshow(cm, interpolation='nearest'); plt.title(name)
    plt.xlabel('Pred'); plt.ylabel('True')
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha='center',va='center')
    out = ART/f"cm_{name}.png"; plt.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out

def log_all(run_id, change, model, metrics, params, cm_path, model_path, code_text):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = RUNS / f"{run_id}_{stamp}"
    rd.mkdir(parents=True, exist_ok=True)
    # save params/metrics/code
    (rd/"params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    mfull = dict(metrics); mfull["timestamp"]=stamp
    (rd/"metrics.json").write_text(json.dumps(mfull, indent=2, ensure_ascii=False), encoding="utf-8")
    (rd/"code.py").write_text(code_text, encoding="utf-8")
    # copy artifacts
    cm_local = ""; model_local = ""
    if cm_path and Path(cm_path).exists():
        cm_local = str(rd/Path(cm_path).name); shutil.copy2(cm_path, cm_local)
    if model_path and Path(model_path).exists():
        model_local = str(rd/Path(model_path).name); shutil.copy2(model_path, model_local)
    # CSV row
    row = {
        "timestamp": stamp, "id": run_id, "change": change, "model": model,
        "acc": metrics.get("acc"), "precision": metrics.get("precision"),
        "recall": metrics.get("recall"), "f1": metrics.get("f1"),
        "run_dir": str(rd), "code_path": str(rd/'code.py'),
        "cm_path": cm_local or str(cm_path or ""), "params_path": str(rd/'params.json'),
        "metrics_path": str(rd/'metrics.json'), "model_path": model_local or str(model_path or ""),
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

# ====== RUN A2 ======
# data
X_train, y_train, _ = load_split("train")
X_val,   y_val,   _ = load_split("val")
X_test,  y_test,  _ = load_split("test")

# scale + PCA
scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train); Xv = scaler.transform(X_val); Xte = scaler.transform(X_test)
pca = PCA(n_components=4, random_state=RNG).fit(Xtr)
Ztr = pca.transform(Xtr); Zv = pca.transform(Xv); Zte = pca.transform(Xte)

# GMM (OCC)
HEALTHY = (y_train==0)
gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=RNG)
gmm.fit(Ztr[HEALTHY])
val_scores = gmm.score_samples(Zv)
ths = np.percentile(val_scores, np.linspace(0.5,99.5,200))
best_f1, best_thr = -1.0, None
for t in ths:
    y_pred = (val_scores < t).astype(int)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    if f1 > best_f1: best_f1, best_thr = f1, t
test_scores = gmm.score_samples(Zte)
y_pred_gmm = (test_scores < best_thr).astype(int)
m_gmm = metrics_of(y_test, y_pred_gmm)
cm_gmm_path = save_cm(m_gmm["cm"], "A2_GMM")
joblib.dump({"scaler":scaler,"pca":pca,"gmm":gmm,"thr":float(best_thr)}, ART/"gmm_A2.joblib")

# MLP tiny
mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam',
                    max_iter=500, random_state=RNG)
mlp.fit(Ztr, y_train)
y_pred_mlp = mlp.predict(Zte)
m_mlp = metrics_of(y_test, y_pred_mlp)
cm_mlp_path = save_cm(m_mlp["cm"], "A2_MLP")
joblib.dump({"scaler":scaler,"pca":pca,"mlp":mlp}, ART/"mlp_A2.joblib")

# log
params = dict(scaler="zscore", pca_dims=4, outlier_pct=0.05, threshold="val-opt",
              sr=SR, win_s=WIN_S, preprocess=[f"preemph_{ALPHA}"], rng=RNG,
              split_sizes=dict(train=len(y_train), val=len(y_val), test=len(y_test)))
code_txt = open(__file__, 'r', encoding='utf-8').read()

log_all("A2-GMM","Pre-emphasis","GMM",
        {k:m_gmm[k] for k in ["acc","precision","recall","f1"]},
        {**params, "model_hparams": dict(n_components=3, cov_type="diag", thr=float(best_thr))},
        cm_gmm_path, ART/"gmm_A2.joblib", code_txt)

log_all("A2-MLP","Pre-emphasis","MLP",
        {k:m_mlp[k] for k in ["acc","precision","recall","f1"]},
        {**params, "model_hparams": dict(hidden=(10,), act="relu", solver="adam", max_iter=500)},
        cm_mlp_path, ART/"mlp_A2.joblib", code_txt)

print("\n=== A2 results ===")
print("GMM:", {k: round(m_gmm[k],3) for k in ["acc","precision","recall","f1"]})
print("MLP:", {k: round(m_mlp[k],3) for k in ["acc","precision","recall","f1"]})
print(f"All saved under: {ART} (CSV: {ART/'variants.csv'})")
