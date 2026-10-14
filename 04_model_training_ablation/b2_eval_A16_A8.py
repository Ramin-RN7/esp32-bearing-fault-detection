# B2 eval: file-level metrics for A8-GMM and A16-MLP on data/segments_split_B2
# Fix: features25 ALWAYS returns 25-D (8 bandpowers), even at 8 kHz (above-Nyquist bands -> 0).
from pathlib import Path
from datetime import datetime
import numpy as np, json, csv, joblib
import soundfile as sf, librosa, matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import skew, kurtosis
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

HERE = Path(__file__).resolve().parent
TEST = HERE / "data" / "segments_split_B2" / "test"
ART  = HERE / "artifacts"; RUNS = ART / "runs"
ART.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
print(f"[PATH] TEST = {TEST}"); print(f"[PATH] ART  = {ART}")

def load16k(fp):
    y, sr = sf.read(fp, dtype='float32', always_2d=False)
    if y.ndim>1: y=y[:,0]
    if sr!=16000: y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    return y

def prefix_of(p):
    n = Path(p).name
    return n.split("__seg")[0] if "__seg" in n else n.rsplit(".",1)[0]

def save_cm(cm, out_png, title):
    fig = plt.figure(figsize=(3.6,3.2), dpi=120); ax = fig.add_subplot(111)
    ax.imshow(cm, interpolation="nearest"); ax.set_title(title)
    ax.set_xticks([0,1]); ax.set_yticks([0,1]); ax.set_xticklabels(["normal","anomaly"]); ax.set_yticklabels(["normal","anomaly"])
    for (i,j),v in np.ndenumerate(cm): ax.text(j,i,str(v), ha='center', va='center')
    fig.tight_layout(); fig.savefig(out_png); plt.close(fig)

def append_row(row):
    vc = ART/"variants.csv"; vq = ART/"variants_queue.csv"
    header = list(row.keys())
    try:
        write_header = not vc.exists()
        with open(vc,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=header)
            if write_header: w.writeheader()
            w.writerow(row)
        print(f"[OK] appended: {vc}")
    except PermissionError:
        qh = not vq.exists()
        with open(vq,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=header)
            if qh: w.writeheader()
            w.writerow(row)
        print(f"[WARN] variants.csv locked → queued: {vq}")

def metrics_of(y, yhat):
    return dict(
        acc=accuracy_score(y,yhat),
        precision=precision_score(y,yhat, zero_division=0),
        recall=recall_score(y,yhat, zero_division=0),
        f1=f1_score(y,yhat, zero_division=0),
        cm=confusion_matrix(y,yhat, labels=[0,1]),
    )

# ---- features (always 25-D) ----
BANDS_8 = [(0,250),(250,500),(500,1000),(1000,2000),(2000,3000),(3000,4000),(4000,6000),(6000,8000)]
def bandpower(y, sr, fmin, fmax):
    S = np.abs(np.fft.rfft(y))**2
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    idx = (freqs>=fmin) & (freqs<fmax)
    if not np.any(idx):  # band above Nyquist -> 0
        return 0.0
    return float(S[idx].sum()/(S.sum()+1e-12))

def spectral_slope(y):
    S = np.abs(np.fft.rfft(y)); x = np.arange(len(S), dtype=np.float32)
    xm, ym = x.mean(), S.mean()
    return float(((x-xm)*(S-ym)).sum()/(((x-xm)**2).sum()+1e-12))

def autocorr_feats(y, max_lag=200):
    x=y-y.mean(); ac = np.correlate(x,x,mode='full')[len(x)-1:len(x)+max_lag]
    if len(ac)<2: return 0.0, 0
    ac[0]=0.0; k=int(np.argmax(ac))
    return float(ac[k]/(np.sum(x**2)+1e-12)), int(k)

def features25(x, sr):
    rms = float(np.sqrt(np.mean(x**2)+1e-12))
    feat = [float(np.mean(x)), float(np.std(x)), rms,
            float(np.max(np.abs(x))), float(np.ptp(x)),
            float(skew(x)), float(kurtosis(x,fisher=True)),
            float(np.mean(np.abs(np.diff(x)))),
            float(np.mean(np.square(np.diff(x)))),
            float(np.count_nonzero(librosa.zero_crossings(x)))/len(x)]
    # always 8 band slots (above-Nyquist become 0 at 8k)
    for f1,f2 in BANDS_8:
        feat.append(bandpower(x, sr, f1, f2))
    S = np.abs(np.fft.rfft(x)); freqs = np.fft.rfftfreq(len(x), 1/sr)
    centroid = float((freqs*(S+1e-12)).sum()/(S.sum()+1e-12))
    bw = float(np.sqrt(((freqs-centroid)**2*(S+1e-12)).sum()/(S.sum()+1e-12)))
    roll = float(librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85).mean())
    flat = float(librosa.feature.spectral_flatness(y=x).mean())
    slope = spectral_slope(x); ac_val, ac_lag = autocorr_feats(x)
    return np.asarray(feat + [centroid, bw, roll, flat, slope, ac_val, float(ac_lag)], dtype=np.float32)

# ---- model-specific preprocessing ----
def bandpass_1_5k(x, sr=16000):
    nyq = 0.5*sr; b,a = butter(4, [1000/nyq, 5000/nyq], btype='bandpass')
    return filtfilt(b,a,x).astype(np.float32)

def envlog8k_feats_from16k(x16):
    x8 = librosa.resample(x16, orig_sr=16000, target_sr=8000)
    need = int(8000*2.0)
    if len(x8)<need: x8 = np.pad(x8,(0,need-len(x8)))
    env = np.abs(hilbert(x8)); x = np.log1p(env).astype(np.float32)
    return features25(x, 8000)  # returns 25-D

# ---- evals ----
def eval_A8():
    mdl = joblib.load(ART/"gmm_A8.joblib")
    scaler, pca, gmm, thr = mdl["scaler"], mdl["pca"], mdl["gmm"], float(mdl["thr"])
    files = [(p,0) for p in sorted((TEST/"normal").glob("*.wav"))] + [(p,1) for p in sorted((TEST/"anomaly").glob("*.wav"))]
    if not files: raise SystemExit(f"[ERR] No WAVs under {TEST}")
    seg_y, seg_hat, prefs = [], [], []
    for fp,y in files:
        x = load16k(fp); need = int(16000*2.0)
        if len(x)<need: x = np.pad(x,(0,need-len(x)))
        x = bandpass_1_5k(x, 16000)
        feat = features25(x, 16000).reshape(1,-1)
        Z = pca.transform(scaler.transform(feat))
        score = float(gmm.score_samples(Z)[0])
        seg_y.append(y); seg_hat.append(int(score<thr)); prefs.append(prefix_of(fp))
    mseg = metrics_of(np.array(seg_y,int), np.array(seg_hat,int))
    byp={}
    for y,yh,pr in zip(seg_y, seg_hat, prefs):
        d=byp.setdefault(pr, {"y":[], "yh":[]}); d["y"].append(y); d["yh"].append(yh)
    y_file,yhat_file=[],[]
    for pr,d in byp.items():
        yt=1 if (np.mean(d["y"])>=0.5) else 0
        yh=1 if (np.mean(d["yh"])>=0.5) else 0
        y_file.append(yt); yhat_file.append(yh)
    mfile = metrics_of(np.array(y_file,int), np.array(yhat_file,int))
    return mseg, mfile

def eval_A16():
    mdl = joblib.load(ART/"mlp_A16.joblib"); scaler, pca, mlp = mdl["scaler"], mdl["pca"], mdl["mlp"]
    files = [(p,0) for p in sorted((TEST/"normal").glob("*.wav"))] + [(p,1) for p in sorted((TEST/"anomaly").glob("*.wav"))]
    if not files: raise SystemExit(f"[ERR] No WAVs under {TEST}")
    seg_y, seg_prob, prefs = [], [], []
    for fp,y in files:
        x16 = load16k(fp)
        feat = envlog8k_feats_from16k(x16).reshape(1,-1)  # now 25-D
        Z = pca.transform(scaler.transform(feat))
        prob = float(mlp.predict_proba(Z)[0,1])
        seg_y.append(y); seg_prob.append(prob); prefs.append(prefix_of(fp))
    seg_hat = (np.array(seg_prob)>=0.5).astype(int)
    mseg = metrics_of(np.array(seg_y,int), seg_hat)
    byp={}
    for y,pr,pref in zip(seg_y, seg_prob, prefs):
        d=byp.setdefault(pref, {"y":[], "p":[]}); d["y"].append(y); d["p"].append(pr)
    y_file,yhat_file=[],[]
    for pref,d in byp.items():
        yt=1 if (np.mean(d["y"])>=0.5) else 0
        yh=1 if (np.mean(d["p"])>=0.5) else 0
        y_file.append(yt); yhat_file.append(yh)
    mfile = metrics_of(np.array(y_file,int), np.array(yhat_file,int))
    return mseg, mfile

if __name__=="__main__":
    # A8
    mseg8, mfile8 = eval_A8()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out8 = RUNS / f"B2-A8-GMM-FILE_{ts}"; out8.mkdir(parents=True, exist_ok=True)
    save_cm(mseg8["cm"], out8/"cm_B2_A8_segment.png", "B2 A8 SEG")
    save_cm(mfile8["cm"], out8/"cm_B2_A8_file.png",    "B2 A8 FILE")
    (out8/"metrics_file.json").write_text(json.dumps({
        "file_precision": float(mfile8["precision"]),
        "file_recall": float(mfile8["recall"]),
        "file_f1": float(mfile8["f1"]),
        "file_acc": float(mfile8["acc"])
    }, indent=2), encoding="utf-8")
    append_row({
        "timestamp": ts,"id":"B2-A8","change":"resegment_B2","model":"GMM-file",
        "acc": float(mfile8["acc"]), "precision": float(mfile8["precision"]),
        "recall": float(mfile8["recall"]), "f1": float(mfile8["f1"]),
        "run_dir": str(out8), "code_path": str(out8/"code.py"),
        "cm_path": str(out8/"cm_B2_A8_file.png"),
        "params_path": "", "metrics_path": str(out8/"metrics_file.json"),
        "model_path": "", "notes": "B2, majority"
    })
    (out8/"code.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    # A16
    mseg16, mfile16 = eval_A16()
    ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
    out16 = RUNS / f"B2-A16-MLP-FILE_{ts2}"; out16.mkdir(parents=True, exist_ok=True)
    save_cm(mseg16["cm"], out16/"cm_B2_A16_segment.png", "B2 A16 SEG")
    save_cm(mfile16["cm"], out16/"cm_B2_A16_file.png",    "B2 A16 FILE")
    (out16/"metrics_file.json").write_text(json.dumps({
        "file_precision": float(mfile16["precision"]),
        "file_recall": float(mfile16["recall"]),
        "file_f1": float(mfile16["f1"]),
        "file_acc": float(mfile16["acc"])
    }, indent=2), encoding="utf-8")
    append_row({
        "timestamp": ts2,"id":"B2-A16","change":"resegment_B2","model":"MLP-file",
        "acc": float(mfile16["acc"]), "precision": float(mfile16["precision"]),
        "recall": float(mfile16["recall"]), "f1": float(mfile16["f1"]),
        "run_dir": str(out16), "code_path": str(out16/"code.py"),
        "cm_path": str(out16/"cm_B2_A16_file.png"),
        "params_path": "", "metrics_path": str(out16/"metrics_file.json"),
        "model_path": "", "notes": "B2, mean(prob)"
    })
    (out16/"code.py").write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")

    print("\n=== B2 results ===")
    print("A8-GMM (file):", {k: round(mfile8[k],3) for k in ["acc","precision","recall","f1"]})
    print("A16-MLP (file):", {k: round(mfile16[k],3) for k in ["acc","precision","recall","f1"]})
