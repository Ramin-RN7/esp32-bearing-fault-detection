# A41: Median-rule file aggregation for A16-MLP and A8-GMM
# Recompute TEST segment predictions from audio, group by prefix.
# File decision:
#   - A16-MLP: anomaly if median(prob_anomaly) >= 0.5
#   - A8-GMM : anomaly if median(score) < thr
# Logs 2 runs: A41-A16 (MLP-file) and A41-A8 (GMM-file). Saves seg/file CMs, metrics.json, params.json, code.py, and appends to variants.csv.

from pathlib import Path
from datetime import datetime
import numpy as np, json, joblib, shutil, csv
import matplotlib.pyplot as plt
import librosa, soundfile as sf
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import skew, kurtosis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---- fixed paths ----
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "segments_split" / "test"
ART  = HERE / "artifacts"
RUNS = ART / "runs"
ART.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
print(f"[PATH] TEST = {DATA}"); print(f"[PATH] ART  = {ART}")

# ---------- helpers ----------
def load_wav(fp):
    y, sr = sf.read(fp, dtype='float32', always_2d=False)
    if y.ndim>1: y = y[:,0]
    return y, sr

def resamp(y, sr, target):
    return y if sr==target else librosa.resample(y, orig_sr=sr, target_sr=target)

def spectral_slope(y):
    S = np.abs(np.fft.rfft(y)); x = np.arange(len(S), dtype=np.float32)
    xm, ym = x.mean(), S.mean()
    num = ((x-xm)*(S-ym)).sum()
    den = ((x-xm)**2).sum()+1e-12
    return float(num/den)

def autocorr_feats(y, max_lag=200):
    x = y - y.mean()
    ac = np.correlate(x, x, mode='full')[len(x)-1:len(x)+max_lag]
    if len(ac)<2: return 0.0, 0
    ac[0]=0.0; k=int(np.argmax(ac))
    return float(ac[k]/(np.sum(x**2)+1e-12)), int(k)

def bandpower(y, sr, fmin, fmax):
    S = np.abs(np.fft.rfft(y))**2
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    idx = (freqs>=fmin)&(freqs<fmax)
    return float(S[idx].sum()/(S.sum()+1e-12))

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
    return np.asarray(feat, dtype=np.float32)  # 25-D

def prefix_of(path_str):
    name = Path(path_str).name
    if "__seg" in name:
        return name.split("__seg")[0]
    return name.rsplit(".",1)[0]

def save_cm(cm, out_png, title):
    fig = plt.figure(figsize=(3.6,3.2), dpi=120)
    ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(["normal","anomaly"]); ax.set_yticklabels(["normal","anomaly"])
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v),ha='center',va='center')
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def metrics_of(y_true, y_pred):
    return dict(
        acc=accuracy_score(y_true,y_pred),
        precision=precision_score(y_true,y_pred, zero_division=0),
        recall=recall_score(y_true,y_pred, zero_division=0),
        f1=f1_score(y_true,y_pred, zero_division=0),
        cm=confusion_matrix(y_true,y_pred, labels=[0,1])
    )

def log_all(run_id, change, model, file_metrics, outdir, file_cm_png, params_extra):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (outdir/"params.json").write_text(json.dumps(params_extra, indent=2, ensure_ascii=False), encoding="utf-8")
    (outdir/"metrics.json").write_text(json.dumps(file_metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    code_txt = Path(__file__).read_text(encoding="utf-8")
    (outdir/"code.py").write_text(code_txt, encoding="utf-8")
    row = {
        "timestamp": stamp, "id": run_id, "change": change, "model": model,
        "acc": file_metrics.get("file_acc"), "precision": file_metrics.get("file_precision"),
        "recall": file_metrics.get("file_recall"), "f1": file_metrics.get("file_f1"),
        "run_dir": str(outdir), "code_path": str(outdir/"code.py"),
        "cm_path": str(file_cm_png), "params_path": str(outdir/"params.json"),
        "metrics_path": str(outdir/"metrics.json"), "model_path": "", "notes": ""
    }
    csv_path = ART/"variants.csv"; write_header = not csv_path.exists()
    with open(csv_path,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print(f"[OK] saved: {outdir}\n[OK] appended: {csv_path}")

# ---------- A8-GMM (bandpass 1–5 kHz @16k, baseline 25D) ----------
def bandpass_1_5k(x, sr=16000):
    nyq = 0.5*sr
    from scipy.signal import butter, filtfilt
    b,a = butter(4, [1000/nyq, 5000/nyq], btype='bandpass')
    return filtfilt(b,a,x).astype(np.float32)

def run_A8_GMM_median():
    mdl_path = ART/"gmm_A8.joblib"
    if not mdl_path.exists():
        print("[ERR] Missing artifacts/gmm_A8.joblib"); return
    bundle = joblib.load(mdl_path)
    scaler, pca, gmm, thr = bundle["scaler"], bundle["pca"], bundle["gmm"], float(bundle["thr"])

    files = [(p,0) for p in sorted((DATA/"normal").glob("*.wav"))] + \
            [(p,1) for p in sorted((DATA/"anomaly").glob("*.wav"))]
    if not files:
        print("[ERR] No test WAVs under", DATA); return

    seg_true, seg_pred, seg_scores, seg_prefix = [], [], [], []
    for fp, y in files:
        ysig, sr = load_wav(fp)
        ysig = resamp(ysig, sr, 16000)
        ysig = bandpass_1_5k(ysig, 16000)
        feat = features25_baseline(ysig, 16000).reshape(1,-1)
        Z = pca.transform(scaler.transform(feat))
        score = float(gmm.score_samples(Z)[0])
        yhat = int(score < thr)  # anomaly if below thr
        seg_true.append(y); seg_pred.append(yhat); seg_scores.append(score); seg_prefix.append(prefix_of(fp))

    seg_true = np.array(seg_true,int); seg_pred = np.array(seg_pred,int)
    mseg = metrics_of(seg_true, seg_pred)

    # FILE decision by MEDIAN score vs thr
    by_pref = {}
    for y,sc,pref in zip(seg_true, seg_scores, seg_prefix):
        d = by_pref.setdefault(pref, {"ys":[], "scores":[]})
        d["ys"].append(int(y)); d["scores"].append(float(sc))
    y_file, yh_file = [], []
    for pref,d in by_pref.items():
        yt = 1 if (np.mean(d["ys"])>=0.5) else 0
        med_score = float(np.median(d["scores"]))
        yhat = 1 if med_score < thr else 0
        y_file.append(yt); yh_file.append(yhat)
    y_file = np.array(y_file,int); yh_file = np.array(yh_file,int)
    mfile = metrics_of(y_file, yh_file)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = RUNS/f"A41-A8_{ts}"; outdir.mkdir(parents=True, exist_ok=True)
    seg_png = outdir/"cm_A41_A8_segment.png"; file_png = outdir/"cm_A41_A8_file.png"
    save_cm(mseg["cm"], seg_png, "A41-A8 SEG (median score)")
    save_cm(mfile["cm"], file_png, "A41-A8 FILE (median score)")

    file_metrics = {
        "seg_acc": float(mseg["acc"]), "seg_precision": float(mseg["precision"]),
        "seg_recall": float(mseg["recall"]), "seg_f1": float(mseg["f1"]),
        "file_acc": float(mfile["acc"]), "file_precision": float(mfile["precision"]),
        "file_recall": float(mfile["recall"]), "file_f1": float(mfile["f1"])
    }
    params = {"base_run":"A8-GMM","rule":"median(score) vs thr","sr":16000,"preprocess":"bandpass_1_5k","features":"baseline25","thr":thr}
    log_all("A41-A8","aggregation_median","GMM-file",
            {"file_acc":file_metrics["file_acc"],"file_precision":file_metrics["file_precision"],
             "file_recall":file_metrics["file_recall"],"file_f1":file_metrics["file_f1"]},
            outdir, file_png, params)
    (outdir/"metrics.json").write_text(json.dumps(file_metrics, indent=2), encoding="utf-8")
    print("=== A41 (A8-GMM) ===")
    print("Segment-level:", {k: round(file_metrics[f"seg_{k}"],3) for k in ["acc","precision","recall","f1"]})
    print("File-level:   ", {k: round(file_metrics[f"file_{k}"],3) for k in ["acc","precision","recall","f1"]})
    print("Saved:", outdir)

# ---------- A16-MLP (Hilbert envelope + log @8k, 25D features on envelope) ----------
def features25_envelope_log_8k(x16k):
    x8k = resamp(x16k, 16000, 8000)
    env = np.abs(hilbert(x8k))
    x = np.log1p(env).astype(np.float32)
    need = int(8000*2.0)
    if len(x) < need:
        x = np.pad(x, (0, need-len(x)))
    return features25_baseline(x, 8000)

def run_A16_MLP_median():
    mdl_path = ART/"mlp_A16.joblib"
    if not mdl_path.exists():
        print("[ERR] Missing artifacts/mlp_A16.joblib"); return
    bundle = joblib.load(mdl_path)
    scaler, pca, mlp = bundle["scaler"], bundle["pca"], bundle["mlp"]

    files = [(p,0) for p in sorted((DATA/"normal").glob("*.wav"))] + \
            [(p,1) for p in sorted((DATA/"anomaly").glob("*.wav"))]
    if not files:
        print("[ERR] No test WAVs under", DATA); return

    seg_true, seg_prob, seg_pred, seg_prefix = [], [], [], []
    for fp, y in files:
        ysig, sr = load_wav(fp)
        if sr != 16000: ysig = resamp(ysig, sr, 16000)
        feat = features25_envelope_log_8k(ysig).reshape(1,-1)
        Z = pca.transform(scaler.transform(feat))
        proba = float(mlp.predict_proba(Z)[0,1])
        yhat = int(proba >= 0.5)
        seg_true.append(y); seg_prob.append(proba); seg_pred.append(yhat); seg_prefix.append(prefix_of(fp))

    seg_true = np.array(seg_true,int); seg_pred = np.array(seg_pred,int)
    mseg = metrics_of(seg_true, seg_pred)

    # FILE decision by MEDIAN probability
    by_pref = {}
    for y,p,pref in zip(seg_true, seg_prob, seg_prefix):
        d = by_pref.setdefault(pref, {"ys":[], "probs":[]})
        d["ys"].append(int(y)); d["probs"].append(float(p))
    y_file, yh_file = [], []
    for pref,d in by_pref.items():
        yt = 1 if (np.mean(d["ys"])>=0.5) else 0
        med_prob = float(np.median(d["probs"]))
        yhat = 1 if med_prob >= 0.5 else 0
        y_file.append(yt); yh_file.append(yhat)
    y_file = np.array(y_file,int); yh_file = np.array(yh_file,int)
    mfile = metrics_of(y_file, yh_file)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = RUNS/f"A41-A16_{ts}"; outdir.mkdir(parents=True, exist_ok=True)
    seg_png = outdir/"cm_A41_A16_segment.png"; file_png = outdir/"cm_A41_A16_file.png"
    save_cm(mseg["cm"], seg_png, "A41-A16 SEG (median prob)")
    save_cm(mfile["cm"], file_png, "A41-A16 FILE (median prob)")

    file_metrics = {
        "seg_acc": float(mseg["acc"]), "seg_precision": float(mseg["precision"]),
        "seg_recall": float(mseg["recall"]), "seg_f1": float(mseg["f1"]),
        "file_acc": float(mfile["acc"]), "file_precision": float(mfile["precision"]),
        "file_recall": float(mfile["recall"]), "file_f1": float(mfile["f1"])
    }
    params = {"base_run":"A16-MLP","rule":"median(prob) >= 0.5","sr":8000,"preprocess":"hilbert_envelope_log","features":"baseline25_on_envelope"}
    log_all("A41-A16","aggregation_median","MLP-file",
            {"file_acc":file_metrics["file_acc"],"file_precision":file_metrics["file_precision"],
             "file_recall":file_metrics["file_recall"],"file_f1":file_metrics["file_f1"]},
            outdir, file_png, params)
    (outdir/"metrics.json").write_text(json.dumps(file_metrics, indent=2), encoding="utf-8")
    print("=== A41 (A16-MLP) ===")
    print("Segment-level:", {k: round(file_metrics[f"seg_{k}"],3) for k in ["acc","precision","recall","f1"]})
    print("File-level:   ", {k: round(file_metrics[f"file_{k}"],3) for k in ["acc","precision","recall","f1"]})
    print("Saved:", outdir)

# ---------- run both ----------
if __name__ == "__main__":
    run_A16_MLP_median()
    run_A8_GMM_median()
