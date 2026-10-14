# A37: Log-Mel (40 bands) features @16 kHz, 2.0 s window
# Feature dim = 40 (log-mel band means) + 5 (centroid, bandwidth, rolloff, flatness, zcr) = 45-D
# Pipeline: StandardScaler -> PCA(8) -> {GMM on healthy + val-tuned thr, MLP(1x10)}
# Saves ONLY under ./artifacts (same logging style as A36).

from pathlib import Path
from datetime import datetime
import numpy as np, json, joblib, shutil
import matplotlib.pyplot as plt
import librosa, soundfile as sf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---- fixed paths ----
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "segments_split"
ART  = HERE / "artifacts"
RUNS = ART / "runs"
ART.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
print(f"[PATH] DATA = {DATA}"); print(f"[PATH] ART  = {ART}")

# ---- constants ----
SR=16000; WIN_S=2.0; RNG=42
N_MELS=40; N_FFT=512; WIN_LEN=400; HOP_LEN=160  # ~25 ms / 10 ms at 16 kHz
PCA_DIMS=8

# ---- io ----
def load_wav(fp):
    y, sr = sf.read(fp, dtype='float32', always_2d=False)
    if y.ndim>1: y = y[:,0]
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y

# ---- features (45) ----
def features_logmel40(y, sr):
    if len(y) < int(SR*WIN_S):
        y = np.pad(y, (0, int(SR*WIN_S)-len(y)))
    x = y.astype(np.float32)

    S = librosa.feature.melspectrogram(
        y=x, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN, win_length=WIN_LEN,
        window='hann', center=True, power=2.0, n_mels=N_MELS
    )
    S_db = librosa.power_to_db(S, ref=np.max)      # log-mel (dB)
    mel_means = S_db.mean(axis=1).astype(np.float32)  # 40 dims

    centroid = float(librosa.feature.spectral_centroid(y=x, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN).mean())
    bandwidth= float(librosa.feature.spectral_bandwidth(y=x, sr=sr, n_fft=N_FFT, hop_length=HOP_LEN).mean())
    rolloff  = float(librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85, n_fft=N_FFT, hop_length=HOP_LEN).mean())
    flatness = float(librosa.feature.spectral_flatness(y=x, n_fft=N_FFT, hop_length=HOP_LEN).mean())
    zcr      = float(librosa.feature.zero_crossing_rate(y=x, frame_length=WIN_LEN, hop_length=HOP_LEN).mean())

    feat = np.concatenate([mel_means, np.array([centroid, bandwidth, rolloff, flatness, zcr], dtype=np.float32)])
    return feat  # 45-D

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
            sig = load_wav(fp)
            X.append(features_logmel40(sig, SR)); y.append(label); paths.append(str(fp))
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
    import csv
    csv_path = ART/"variants.csv"; write_header = not csv_path.exists()
    with open(csv_path,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print(f"[OK] saved: {rd}\n[OK] appended: {csv_path}")

# ==== run ====
from sklearn.metrics import f1_score
X_train, y_train, _ = load_split("train")
X_val,   y_val,   _ = load_split("val")
X_test,  y_test,  _ = load_split("test")

scaler = StandardScaler().fit(X_train)
Xtr = scaler.transform(X_train); Xv = scaler.transform(X_val); Xte = scaler.transform(X_test)
pca = PCA(n_components=PCA_DIMS, random_state=RNG).fit(Xtr)
Ztr = pca.transform(Xtr); Zv = pca.transform(Xv); Zte = pca.transform(Xte)

HEALTHY = (y_train==0)
gmm = GaussianMixture(n_components=3, covariance_type="diag", random_state=RNG).fit(Ztr[HEALTHY])
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
cm_gmm_path = save_cm(m_gmm["cm"], "A37_GMM")
joblib.dump({"scaler":scaler,"pca":pca,"gmm":gmm,"thr":float(best_thr)}, ART/"gmm_A37.joblib")

mlp = MLPClassifier(hidden_layer_sizes=(10,), activation='relu', solver='adam',
                    max_iter=500, random_state=RNG)
mlp.fit(Ztr, y_train)
y_pred_mlp = mlp.predict(Zte)
m_mlp = metrics_of(y_test, y_pred_mlp)
cm_mlp_path = save_cm(m_mlp["cm"], "A37_MLP")
joblib.dump({"scaler":scaler,"pca":pca,"mlp":mlp}, ART/"mlp_A37.joblib")

params = dict(
    sr=SR, win_s=WIN_S, rng=RNG, scaler="zscore", pca_dims=PCA_DIMS, threshold="val-opt",
    features="logmel40_means+centroid+bw+rolloff+flatness+zcr (45D->PCA8)",
    mel=dict(n_mels=N_MELS, n_fft=N_FFT, win_len=WIN_LEN, hop_len=HOP_LEN, log="dB"),
    split_sizes=dict(train=len(y_train), val=len(y_val), test=len(y_test))
)
code_txt = open(__file__, 'r', encoding='utf-8').read()

log_all("A37-GMM","Log-Mel 40 bands (means + 5 stats)","GMM",
        {k:m_gmm[k] for k in ["acc","precision","recall","f1"]},
        {**params,"model_hparams": dict(n_components=3, cov_type="diag", thr=float(best_thr))},
        cm_gmm_path, ART/"gmm_A37.joblib", code_txt)

log_all("A37-MLP","Log-Mel 40 bands (means + 5 stats)","MLP",
        {k:m_mlp[k] for k in ["acc","precision","recall","f1"]},
        {**params,"model_hparams": dict(hidden=(10,), act="relu", solver="adam", max_iter=500)},
        cm_mlp_path, ART/"mlp_A37.joblib", code_txt)

print("\n=== A37 results ===")
print("GMM:", {k: round(m_gmm[k],3) for k in ["acc","precision","recall","f1"]})
print("MLP:", {k: round(m_mlp[k],3) for k in ["acc","precision","recall","f1"]})
print(f"All saved under: {ART}  (CSV: {ART/'variants.csv'})")
