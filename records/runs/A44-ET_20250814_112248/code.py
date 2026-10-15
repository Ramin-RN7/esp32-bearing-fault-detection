# A44: Train + export ExtraTrees (n=15, depth=6) on A16 features (env→log @8 kHz, MinMax, PCA8)
# - Trains on data/segments_split (B0)
# - Saves run bundle under artifacts/runs/A44-ET_YYYYMMDD_HHMMSS
# - Exports emlearn header + preprocess.json to artifacts/exports/A44_ET/
# - Writes segment-level + file-level metrics, CMs, and test_segment_scores.csv

from pathlib import Path
from datetime import datetime
import numpy as np, json, csv, shutil, joblib, sys
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
DATA = HERE / "data" / "segments_split"         # baseline segments (B0)
ART  = HERE / "artifacts"
RUNS = ART / "runs"
EXPO = ART / "exports" / "A44_ET"
for p in (ART, RUNS, EXPO): p.mkdir(parents=True, exist_ok=True)
print(f"[PATH] DATA = {DATA}"); print(f"[PATH] ART  = {ART}")

# ---- constants ----
SR=8000; WIN_S=2.0; RNG=42
PCA_DIMS=8
N_TREES=15; MAX_DEPTH=6

# ---- io ----
def load_wav(fp):
    y, sr = sf.read(fp, dtype='float32', always_2d=False)
    if y.ndim>1: y = y[:,0]
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y.astype(np.float32)

# ---- preprocess: envelope → log ----
def env_log(y: np.ndarray):
    y = y.astype(np.float32)
    env = np.abs(hilbert(y)).astype(np.float32)
    x = np.log(env + 1e-8).astype(np.float32)
    return x

# ---- features (25)  — keep 25-D even at 8 kHz (hi bands become ~0) ----
def bandpower(y, sr, fmin, fmax):
    S = np.abs(np.fft.rfft(y))**2
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    idx = (freqs>=fmin)&(freqs<fmax)
    den = S.sum()+1e-12
    num = S[idx].sum() if np.any(idx) else 0.0
    return float(num/den)

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
    # always 8 bands to keep 25-D (last two zero at 8 kHz)
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

FEAT_NAMES = [
    "mean","std","rms","maxabs","ptp","skew","kurt","mean_abs_diff","mean_sq_diff","zcr",
    "bp_0_250","bp_250_500","bp_500_1000","bp_1000_2000","bp_2000_3000","bp_3000_4000","bp_4000_6000","bp_6000_8000",
    "spec_centroid","spec_bw","rolloff","flatness","slope","ac_peak","ac_lag"
]

# ---- data loader ----
def load_split(split):
    base = DATA / split
    normals  = sorted((base/"normal").glob("*.wav"))
    anomalies= sorted((base/"anomaly").glob("*.wav"))
    if not normals and not anomalies:
        raise FileNotFoundError(f"No WAVs under {base}/(normal|anomaly)")
    X,y,paths,prefixes = [],[],[],[]
    for files,label in [(normals,0),(anomalies,1)]:
        for fp in files:
            sig = env_log(load_wav(fp))
            X.append(features25(sig, SR)); y.append(label); paths.append(str(fp))
            prefixes.append(Path(fp).stem.split("__seg")[0])
    return np.vstack(X), np.array(y), paths, prefixes

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

def log_all(run_id, change, model, metrics, params, cm_path_seg, cm_path_file, model_path, code_text):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = RUNS / f"{run_id}_{stamp}"; rd.mkdir(parents=True, exist_ok=True)
    (rd/"params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    mfull = dict(metrics); mfull["timestamp"]=stamp
    (rd/"metrics.json").write_text(json.dumps(mfull, indent=2, ensure_ascii=False), encoding="utf-8")
    (rd/"code.py").write_text(code_text, encoding="utf-8")
    cm_seg_local=cm_file_local=model_local=""
    if cm_path_seg and Path(cm_path_seg).exists(): cm_seg_local=str(rd/Path(cm_path_seg).name); shutil.copy2(cm_path_seg, cm_seg_local)
    if cm_path_file and Path(cm_path_file).exists(): cm_file_local=str(rd/Path(cm_path_file).name); shutil.copy2(cm_path_file, cm_file_local)
    if model_path and Path(model_path).exists(): model_local=str(rd/Path(model_path).name); shutil.copy2(model_path, model_local)
    row = {"timestamp": stamp,"id": run_id,"change": change,"model": "ET",
           "acc": metrics.get("seg_acc"),"precision": metrics.get("seg_precision"),
           "recall": metrics.get("seg_recall"),"f1": metrics.get("seg_f1"),
           "run_dir": str(rd),"code_path": str(rd/'code.py'),
           "cm_path": cm_seg_local or str(cm_path_seg or ""), "params_path": str(rd/'params.json'),
           "metrics_path": str(rd/'metrics.json'),"model_path": model_local or str(model_path or ""), "notes": f"fileF1={metrics.get('file_f1'):.3f}"}
    csv_path = ART/"variants.csv"; write_header = not csv_path.exists()
    try:
        with open(csv_path,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header: w.writeheader()
            w.writerow(row)
    except PermissionError:
        q = ART/"variants_queue.csv"
        fexists = q.exists()
        with open(q,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(row.keys()))
            if not fexists: w.writeheader()
            w.writerow(row)
        print(f"[WARN] variants.csv locked; appended to: {q}")
    print(f"[OK] saved: {rd}\n[OK] appended: {ART/'variants.csv'}")

# ---- train/eval/export ----
# load data
X_train, y_train, p_tr, pref_tr = load_split("train")
X_val,   y_val,   p_va, pref_va = load_split("val")
X_test,  y_test,  p_te, pref_te = load_split("test")

# scaler + PCA
scaler = MinMaxScaler().fit(X_train)
Xtr = scaler.transform(X_train); Xv = scaler.transform(X_val); Xte = scaler.transform(X_test)
pca = PCA(n_components=PCA_DIMS, random_state=RNG).fit(Xtr)
Ztr = pca.transform(Xtr); Zv = pca.transform(Xv); Zte = pca.transform(Xte)

# model (small ET)
et = ExtraTreesClassifier(n_estimators=N_TREES, max_depth=MAX_DEPTH, random_state=RNG)
et.fit(Ztr, y_train)

# segment metrics (TEST)
probs = et.predict_proba(Zte)[:,1]
y_pred_seg = (probs >= 0.5).astype(int)
m_seg = metrics_of(y_test, y_pred_seg)

# save per-segment scores (for audit/aggregation)
scores_csv = EXPO/"test_segment_scores.csv"
with open(scores_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["prefix","y_true","score_or_prob","kind"])
    w.writeheader()
    for pref, yt, pr in zip(pref_te, y_test, probs):
        w.writerow({"prefix": pref, "y_true": int(yt), "score_or_prob": float(pr), "kind": "prob_anomaly_high"})

# file-level (mean prob ≥ 0.5)
from collections import defaultdict
G = defaultdict(lambda: {"ys":[], "probs":[]})
for pref, yt, pr in zip(pref_te, y_test, probs):
    G[pref]["ys"].append(int(yt)); G[pref]["probs"].append(float(pr))
y_true_file, y_pred_file = [], []
for pref,g in G.items():
    yt = 1 if (np.mean(g["ys"])>=0.5) else 0
    yhat = 1 if (np.mean(g["probs"])>=0.5) else 0
    y_true_file.append(yt); y_pred_file.append(yhat)
y_true_file = np.array(y_true_file, dtype=int); y_pred_file = np.array(y_pred_file, dtype=int)

# metrics dict
from sklearn.metrics import precision_recall_fscore_support
p_file, r_file, f1_file, _ = precision_recall_fscore_support(y_true_file, y_pred_file, average="binary", zero_division=0)
acc_file = accuracy_score(y_true_file, y_pred_file)

metrics = {
    "seg_acc": float(m_seg["acc"]), "seg_precision": float(m_seg["precision"]), "seg_recall": float(m_seg["recall"]), "seg_f1": float(m_seg["f1"]),
    "file_acc": float(acc_file), "file_precision": float(p_file), "file_recall": float(r_file), "file_f1": float(f1_file)
}

# save CMs
cm_seg_path  = save_cm(m_seg["cm"], "A44_ET_segment")
cm_file_path = save_cm(confusion_matrix(y_true_file, y_pred_file), "A44_ET_file")

# save sklearn model (for backup) + export bundle
joblib.dump({"scaler":scaler,"pca":pca,"et":et}, ART/"et_A44.joblib")

# export preprocess.json
preprocess = {
    "pipeline": "A16_envlog_minmax_pca8",
    "sr": SR, "win_s": WIN_S,
    "feature_names": FEAT_NAMES,
    "scaler": {"type":"minmax", "min_": scaler.data_min_.tolist(), "max_": scaler.data_max_.tolist(),
               "scale_": scaler.scale_.tolist(), "min_offset_": scaler.min_.tolist()},
    "pca": {"n_components": int(PCA_DIMS), "mean": pca.mean_.tolist(), "components": pca.components_.tolist()}
}
(EXPO/"preprocess.json").write_text(json.dumps(preprocess, indent=2), encoding="utf-8")

# emlearn export (if available)
header_path = EXPO/"model_et.h"
emlearn_ok = False
try:
    try:
        from emlearn import convert as eml_convert  # common API
    except Exception:
        eml_convert = None
    if eml_convert is not None:
        eml_model = eml_convert.convert_classifier(et, method='inline')
        # emlearn ≥0.20: to_c(); older: save()
        code = None
        if hasattr(eml_model, "to_c"):
            code = eml_model.to_c(name="A44_ET")
            header_path.write_text(code, encoding="utf-8")
            emlearn_ok = True
        elif hasattr(eml_model, "save"):
            eml_model.save(str(header_path))
            emlearn_ok = True
except Exception as e:
    print(f"[WARN] emlearn export failed: {e}")

# record sizes
sizes = {"header_bytes": int(header_path.stat().st_size) if header_path.exists() else 0}
(EXPO/"size.json").write_text(json.dumps(sizes, indent=2), encoding="utf-8")

# ---- log & finish ----
params = dict(scaler="minmax", pca_dims=PCA_DIMS, sr=SR, win_s=WIN_S,
              preprocess=["hilbert_envelope_log"], rng=RNG,
              model_hparams=dict(model="ET", n_estimators=N_TREES, max_depth=MAX_DEPTH),
              export=dict(emlearn_ok=bool(emlearn_ok), header=str(header_path)))
code_txt = open(__file__, 'r', encoding='utf-8').read()

log_all("A44-ET","ET(15,depth6) on A16 features","ET",
        metrics, params, cm_seg_path, cm_file_path, ART/"et_A44.joblib", code_txt)

print("\n=== A44 results (ET-small on A16 features) ===")
print("Segment:", {k: round(metrics[f"seg_{k}"],3) for k in ["acc","precision","recall","f1"]})
print("File:   ", {k: round(metrics[f"file_{k}"],3) for k in ["acc","precision","recall","f1"]})
print(f"Export: emlearn_header={'OK' if emlearn_ok else 'NO'}  -> {header_path if header_path.exists() else '(not written)'}")
print(f"Scores: {scores_csv}")
print(f"All saved under: {ART}  (CSV: {ART/'variants.csv'})")
