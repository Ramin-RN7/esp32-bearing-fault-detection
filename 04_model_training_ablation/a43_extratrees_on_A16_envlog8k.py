# A43: ExtraTrees (50 trees, max_depth=8) on A16 preprocess (Hilbert envelope + log @8k)
# MinMaxScaler -> PCA(8) -> ExtraTrees(50, depth<=8). Saves ONLY under ./artifacts.
# Queue fallback if artifacts/variants.csv is locked.

from pathlib import Path
from datetime import datetime
import numpy as np, json, joblib, shutil, csv
import matplotlib.pyplot as plt
import librosa, soundfile as sf
from scipy.signal import hilbert
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---- paths ----
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "segments_split"
ART  = HERE / "artifacts"
RUNS = ART / "runs"
ART.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
print(f"[PATH] DATA = {DATA}"); print(f"[PATH] ART  = {ART}")

# ---- consts ----
SR_FEAT=8000; WIN_S=2.0; RNG=42
PCA_DIMS=8

# ---- io ----
def load_wav(fp):
    y, sr = sf.read(fp, dtype='float32', always_2d=False)
    if y.ndim>1: y = y[:,0]
    if sr != SR_FEAT: y = librosa.resample(y, orig_sr=sr, target_sr=SR_FEAT)
    return y

# ---- features (baseline25 on envelope-log @8k) ----
def bandpower(y, sr, fmin, fmax):
    S = np.abs(np.fft.rfft(y))**2
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    idx = (freqs>=fmin)&(freqs<fmax)
    return float(S[idx].sum()/(S.sum()+1e-12))

def spectral_slope(y):
    S = np.abs(np.fft.rfft(y)); x = np.arange(len(S), dtype=np.float32)
    xm, ym = x.mean(), S.mean()
    return float(((x-xm)*(S-ym)).sum()/(((x-xm)**2).sum()+1e-12))

def autocorr_feats(y, max_lag=200):
    x = y - y.mean()
    ac = np.correlate(x, x, mode='full')[len(x)-1:len(x)+max_lag]
    if len(ac)<2: return 0.0, 0
    ac[0]=0.0; k=int(np.argmax(ac))
    return float(ac[k]/(np.sum(x**2)+1e-12)), int(k)

def features25_baseline(x, sr):
    rms = float(np.sqrt(np.mean(x**2)+1e-12))
    feat = [
        float(np.mean(x)), float(np.std(x)), rms,
        float(np.max(np.abs(x))), float(np.ptp(x)),
        float(skew(x)), float(kurtosis(x,fisher=True)),
        float(np.mean(np.abs(np.diff(x)))),
        float(np.mean(np.square(np.diff(x)))),
        float(np.count_nonzero(librosa.zero_crossings(x)))/len(x)
    ]
    # up to Nyquist(4k at 8k SR)
    bands=[(0,250),(250,500),(500,1000),(1000,2000),(2000,3000),(3000,4000)]
    for f1,f2 in bands: feat.append(bandpower(x, sr, f1, f2))
    S = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(len(x), 1/sr)
    centroid = float((freqs*(S+1e-12)).sum()/(S.sum()+1e-12))
    bw = float(np.sqrt(((freqs-centroid)**2*(S+1e-12)).sum()/(S.sum()+1e-12)))
    roll = float(librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85).mean())
    flat = float(librosa.feature.spectral_flatness(y=x).mean())
    slope = spectral_slope(x)
    ac_val, ac_lag = autocorr_feats(x)
    feat += [centroid, bw, roll, flat, slope, ac_val, float(ac_lag)]
    return np.asarray(feat, dtype=np.float32)  # 25-D

def features25_envlog8k(y):
    need = int(SR_FEAT*WIN_S)
    if len(y) < need: y = np.pad(y, (0, need-len(y)))
    env = np.abs(hilbert(y))
    x = np.log1p(env).astype(np.float32)
    return features25_baseline(x, SR_FEAT)

# ---- data ----
def load_split(split):
    base = DATA / split
    normals  = sorted((base/"normal").glob("*.wav"))
    anomalies= sorted((base/"anomaly").glob("*.wav"))
    if not normals and not anomalies:
        raise FileNotFoundError(f"No WAVs under {base}/(normal|anomaly)")
    X,y,paths=[],[],[]
    for files,label in [(normals,0),(anomalies,1)]:
        for fp in files:
            sig = load_wav(fp)
            X.append(features25_envlog8k(sig)); y.append(label); paths.append(str(fp))
    return np.vstack(X), np.array(y), paths

# ---- utils ----
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
    csv_path = ART/"variants.csv"; vq_path = ART/"variants_queue.csv"
    write_header = not csv_path.exists()
    try:
        with open(csv_path,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header: w.writeheader()
            w.writerow(row)
        print(f"[OK] saved: {rd}\n[OK] appended: {csv_path}")
    except PermissionError:
        q_header = not vq_path.exists()
        with open(vq_path,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(row.keys()))
            if q_header: w.writeheader()
            w.writerow(row)
        print(f"[WARN] variants.csv locked (likely open). Appended to queue: {vq_path}\n[OK] saved: {rd}")

# ---- run ----
from sklearn.metrics import f1_score
X_train, y_train, _ = load_split("train")
X_val,   y_val,   _ = load_split("val")
X_test,  y_test,  _ = load_split("test")

scaler = MinMaxScaler().fit(X_train)
Xtr = scaler.transform(X_train); Xv = scaler.transform(X_val); Xte = scaler.transform(X_test)
pca = PCA(n_components=PCA_DIMS, random_state=RNG).fit(Xtr)
Ztr = pca.transform(Xtr); Zv = pca.transform(Xv); Zte = pca.transform(Xte)

et = ExtraTreesClassifier(n_estimators=50, max_depth=8, random_state=RNG, n_jobs=-1, bootstrap=False)
et.fit(Ztr, y_train)
y_pred = et.predict(Zte)
m_et = metrics_of(y_test, y_pred)
cm_et_path = save_cm(m_et["cm"], "A43_ET")
joblib.dump({"scaler":scaler,"pca":pca,"et":et}, ART/"et_A43.joblib")

params = dict(
    sr=SR_FEAT, win_s=WIN_S, rng=RNG, scaler="minmax", pca_dims=PCA_DIMS,
    preprocess="hilbert_envelope_log @8k", features="baseline25_on_envelope",
    model_hparams=dict(n_estimators=50, max_depth=8, bootstrap=False)
)
code_txt = open(__file__, 'r', encoding='utf-8').read()

log_all("A43-ET","ET(50,depth8) on A16 preprocess","ET",
        {k:m_et[k] for k in ["acc","precision","recall","f1"]},
        params, cm_et_path, ART/"et_A43.joblib", code_txt)

print("\n=== A43 results (ET on A16) ===")
print({k: round(m_et[k],3) for k in ["acc","precision","recall","f1"]})
print(f"All saved under: {ART}  (CSV: {ART/'variants.csv'})")
