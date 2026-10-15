# E3: Env→Log @8 kHz, MinMax, PCA(8), GMM with 10% healthy outlier removal (refit), VAL-tuned threshold → TEST
# Saves ONLY under ./artifacts, appends a row to variants.csv (same header), and writes a run bundle.

from pathlib import Path
from datetime import datetime
import numpy as np, json, csv, shutil, joblib
import matplotlib.pyplot as plt
import librosa, soundfile as sf
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---- fixed paths ----
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "segments_split"        # baseline (B0)
ART  = HERE / "artifacts"
RUNS = ART / "runs"
ART.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
print(f"[PATH] DATA = {DATA}"); print(f"[PATH] ART  = {ART}")

# ---- constants ----
SR=8000; WIN_S=2.0; RNG=42
PCA_DIMS=8
OUTLIER_PCT=0.10  # 10% trim on healthy for GMM (refit)

# ---- io ----
def load_wav(fp):
    y, sr = sf.read(fp, dtype='float32', always_2d=False)
    if y.ndim>1: y = y[:,0]
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype(np.float32)

# ---- preprocess: envelope → log ----
def env_log(y: np.ndarray):
    y = y.astype(np.float32)
    # Hilbert envelope
    env = np.abs(hilbert(y)).astype(np.float32)
    # log envelope (small epsilon to avoid -inf)
    x = np.log(env + 1e-8).astype(np.float32)
    return x

# ---- features (25) ----
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
    # pad to fixed length
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
    bands=[(0,250),(250,500),(500,1000),(1000,2000),(2000,3000),(3000,4000)]
    # at 8 kHz Nyquist is 4 kHz; keep bands up to 4 kHz (6 bands)
    for f1,f2 in bands: feat.append(bandpower(x, sr, f1, f2))
    S = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(len(x), 1/sr)
    centroid = float((freqs*(S+1e-12)).sum()/(S.sum()+1e-12))
    bw = float(np.sqrt(((freqs-centroid)**2*(S+1e-12)).sum()/(S.sum()+1e-12)))
    roll = float(librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85).mean())
    flat = float(librosa.feature.spectral_flatness(y=x).mean())
    slope = spectral_slope(x)
    ac_val, ac_lag = autocorr_feats(x)
    feat += [centroid, bw, roll, flat, slope, ac_val, float(ac_lag)]
    return np.asarray(feat, dtype=np.float32)  # total 25

# ---- data loader ----
def load_split(split):
    base = DATA / split
    normals  = sorted((base/"normal").glob("*.wav"))
    anomalies= sorted((base/"anomaly").glob("*.wav"))
    if not normals and not anomalies:
        raise FileNotFoundError(f"No WAVs under {base}/(normal|anomaly)")
    X,y,paths = [],[],[]
    for files,label in [(normals,0),(anomalies,1)]:
        for fp in files:
            sig = env_log(load_wav(fp))
            X.append(features25(sig, SR)); y.append(label); paths.append(str(fp))
    return np.vstack(X), np.array(y), paths

# ---- metrics / logging ----
def metrics_of(y_true, y_pred):
    return dict(
        acc=accuracy_score(y_true,y_pred),
        precision=precision_score(y_true,y_pred, zero_division=0),
        recall=recall_score(y_true,y_pred, zero_division=0),
        f1=f1_score(y_true,y_pred, zero_division=0),
        cm=confusion_matrix(y_true,y_pred)
    )

def save_cm(cm, name):
    fig = plt.figure(); plt.imshow(cm, interpolation='nearest')
    plt.title(name); plt.xlabel('Pred'); plt.ylabel('True')
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha='center',va='center')
    out = ART/f"cm_{name}.png"; plt.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out

def log_all(run_id, change, model, metrics, params, cm_path, model_path, code_text):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = RUNS / f"{run_id}_{stamp}"; rd.mkdir(parents=True, exist_ok=True)
    (rd/"params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    mfull = dict(metrics); mfull["timestamp"]=stamp
    (rd/"metrics.json").write_text(json.dumps(mfull, indent=2, ensure_ascii=False), encoding="utf-8")
    (rd/"code.py").write_text(code_text, encoding="utf-8")
    cm_local=model_local=""
    if cm_path and Path(cm_path).exists(): cm_local=str(rd/Path(cm_path).name); shutil.copy2(cm_path, cm_local)
    if model_path and Path(model_path).exists(): model_local=str(rd/Path(model_path).name); shutil.copy2(model_path, model_local)
    row = {"timestamp": stamp,"id": run_id,"change": change,"model": model,
           "acc": metrics.get("acc"),"precision": metrics.get("precision"),
           "recall": metrics.get("recall"),"f1": metrics.get("f1"),
           "run_dir": str(rd),"code_path": str(rd/'code.py'),
           "cm_path": cm_local or str(cm_path or ""), "params_path": str(rd/'params.json'),
           "metrics_path": str(rd/'metrics.json'),"model_path": model_local or str(model_path or ""), "notes": ""}
    csv_path = ART/"variants.csv"; write_header = not csv_path.exists()
    try:
        with open(csv_path,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header: w.writeheader()
            w.writerow(row)
    except PermissionError:
        # fallback queue if variants.csv open
        q = ART/"variants_queue.csv"
        fexists = q.exists()
        with open(q,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(row.keys()))
            if not fexists: w.writeheader()
            w.writerow(row)
        print(f"[WARN] variants.csv locked; appended to: {q}")
    print(f"[OK] saved: {rd}\n[OK] appended: {ART/'variants.csv'}")

# ==== run ====
from sklearn.metrics import f1_score

# load data
X_train, y_train, _ = load_split("train")
X_val,   y_val,   _ = load_split("val")
X_test,  y_test,  _ = load_split("test")

# scale + PCA
scaler = MinMaxScaler().fit(X_train)
Xtr = scaler.transform(X_train); Xv = scaler.transform(X_val); Xte = scaler.transform(X_test)
pca = PCA(n_components=PCA_DIMS, random_state=RNG).fit(Xtr)
Ztr = pca.transform(Xtr); Zv = pca.transform(Xv); Zte = pca.transform(Xte)

HEALTHY = (y_train==0)
Z_h = Ztr[HEALTHY]

# --- provisional fit + trim lowest 10% by likelihood, then refit ---
gmm0 = GaussianMixture(n_components=3, covariance_type="diag", random_state=RNG).fit(Z_h)
scores_h = gmm0.score_samples(Z_h)  # higher = more normal
thr_keep = np.percentile(scores_h, 100*(OUTLIER_PCT))  # drop bottom 10%
keep = scores_h > thr_keep
Z_h_keep = Z_h[keep]
gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=RNG).fit(Z_h_keep)

# --- threshold tuning on VAL ---
val_scores = gmm.score_samples(Zv)
ths = np.percentile(val_scores, np.linspace(0.5,99.5,200))
best_f1, best_thr = -1.0, None
for t in ths:
    y_pred = (val_scores < t).astype(int)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    if f1 > best_f1: best_f1, best_thr = f1, t

# --- TEST ---
test_scores = gmm.score_samples(Zte)
y_pred_gmm = (test_scores < best_thr).astype(int)
m_gmm = metrics_of(y_test, y_pred_gmm)
cm_gmm_path = save_cm(m_gmm["cm"], "E3_GMM")
model_path = ART/"gmm_E3.joblib"
joblib.dump({"scaler":scaler,"pca":pca,"gmm":gmm,"thr":float(best_thr),
             "outlier_pct": OUTLIER_PCT, "kept_frac": float(len(Z_h_keep)/max(len(Z_h),1))},
            model_path)

# --- log ---
params = dict(scaler="minmax", pca_dims=PCA_DIMS, outlier_pct=OUTLIER_PCT, threshold="val-opt",
              sr=SR, win_s=WIN_S, preprocess=["hilbert_envelope_log"], rng=RNG,
              split_sizes=dict(train=len(y_train), val=len(y_val), test=len(y_test)))
code_txt = open(__file__, 'r', encoding='utf-8').read()

log_all("E3-GMM","Env→Log@8k + MinMax + PCA8 + GMM (trim10% healthy, refit)","GMM",
        {k:m_gmm[k] for k in ["acc","precision","recall","f1"]},
        {**params,"model_hparams": dict(n_components=3, cov_type="diag", thr=float(best_thr))},
        cm_gmm_path, model_path, code_txt)

print("\n=== E3 results (GMM, trim10%) ===")
print("GMM:", {k: round(m_gmm[k],3) for k in ["acc","precision","recall","f1"]})
print(f"Kept healthy frac: {len(Z_h_keep)}/{len(Z_h)} = {len(Z_h_keep)/max(1,len(Z_h)):.3f}")
print(f"All saved under: {ART}  (CSV: {ART/'variants.csv'})")
