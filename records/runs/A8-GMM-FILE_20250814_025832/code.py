# A35a: Per-file aggregation for A8-GMM (Band-pass 1–5 kHz, 25-D, PCA(4), GMM)
# - Loads saved model: artifacts/gmm_A8.joblib
# - Re-scores TEST segments, groups by "<prefix>__seg####.wav" prefix
# - Segment-level CSV + metrics
# - File-level (per-recording) CSV + metrics + confusion matrix
# - Writes a short report.txt explaining why/what/how and why it matters
# - Appends a row to artifacts/variants.csv and saves everything under artifacts/runs/

from pathlib import Path
from datetime import datetime
import json, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
import scipy.signal as sig
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# ---- fixed paths ----
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "segments_split" / "test"
ART  = HERE / "artifacts"
RUNS = ART / "runs"
ART.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
print(f"[PATH] TEST = {DATA}"); print(f"[PATH] ART  = {ART}")

# ---- constants (match A8) ----
SR = 16000
WIN_S = 2.0
BP_LOW = 1000
BP_HIGH = 5000
RNG = 42

MODEL_PATH = ART / "gmm_A8.joblib"   # must exist (from your A8 run)
RUN_ID = "A8-GMM-FILE"
CHANGE = "Per-file aggregation of A8 (BP1–5k) GMM scores (mean score vs thr)"
MODEL_NAME = "GMM-file"

# ---- helpers ----
def load_wav(fp):
    y, sr = sf.read(fp, dtype='float32', always_2d=False)
    if y.ndim>1: y = y[:,0]
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y

def bandpass_1_5k(y, sr=SR, f1=BP_LOW, f2=BP_HIGH):
    sos = sig.iirfilter(N=4, Wn=(f1, f2), btype='bandpass', fs=sr, output='sos')
    return sig.sosfiltfilt(sos, y).astype(np.float32)

# 25 handcrafted features (same family used across A0–A8)
from scipy.stats import skew, kurtosis
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

def group_id_from_fp(fp: Path) -> str:
    # Group by prefix before "__seg####"
    return fp.stem.split("__seg")[0]

def metrics_of(y_true, y_pred):
    return dict(
        acc=accuracy_score(y_true,y_pred),
        precision=precision_score(y_true,y_pred, zero_division=0),
        recall=recall_score(y_true,y_pred, zero_division=0),
        f1=f1_score(y_true,y_pred, zero_division=0),
        cm=confusion_matrix(y_true,y_pred)
    )

def save_cm(cm, name, outdir: Path):
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title(name); plt.xlabel('Pred'); plt.ylabel('True')
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha='center',va='center')
    out = outdir/f"cm_{name}.png"; plt.tight_layout(); fig.savefig(out, dpi=160); plt.close(fig)
    return out

def log_all(run_id, change, model, metrics_row, params, cm_path, code_text, extras: dict):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    rd = RUNS / f"{run_id}_{stamp}"; rd.mkdir(parents=True, exist_ok=True)
    # write basics
    (rd/"params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    mfull = dict(metrics_row); mfull["timestamp"]=stamp
    (rd/"metrics.json").write_text(json.dumps(mfull, indent=2, ensure_ascii=False), encoding="utf-8")
    (rd/"code.py").write_text(code_text, encoding="utf-8")
    # copy cm
    cm_local=""
    if cm_path and Path(cm_path).exists():
        cm_local=str(rd/Path(cm_path).name); shutil.copy2(cm_path, cm_local)
    # write extras (csvs, report)
    for k, p in extras.items():
        if isinstance(p, Path) and p.exists():
            shutil.copy2(p, rd/p.name)
    # append variants.csv
    row = {"timestamp": stamp,"id": run_id,"change": change,"model": model,
           "acc": metrics_row.get("acc"),"precision": metrics_row.get("precision"),
           "recall": metrics_row.get("recall"),"f1": metrics_row.get("f1"),
           "run_dir": str(rd),"code_path": str(rd/'code.py'),
           "cm_path": cm_local or str(cm_path or ""), "params_path": str(rd/'params.json'),
           "metrics_path": str(rd/'metrics.json'),"model_path": "", "notes": "Per-file aggregation"}
    import csv
    csv_path = ART/"variants.csv"; write_header = not csv_path.exists()
    with open(csv_path,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)
    print(f"[OK] saved: {rd}\n[OK] appended: {csv_path}")
    return rd

# ---- load model ----
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH} (run A8 first)")
bundle = joblib.load(MODEL_PATH)
scaler = bundle["scaler"]; pca = bundle["pca"]; gmm = bundle["gmm"]; thr = float(bundle["thr"])

# ---- collect TEST files ----
normal_dir  = DATA / "normal"
anomaly_dir = DATA / "anomaly"
normals  = sorted(normal_dir.glob("*.wav"))
anomalies= sorted(anomaly_dir.glob("*.wav"))
if not normals and not anomalies:
    raise FileNotFoundError(f"No WAVs under {DATA}/(normal|anomaly)")
files_and_labels = [(fp,0) for fp in normals] + [(fp,1) for fp in anomalies]

# ---- segment-level scoring ----
rows = []
for fp, y in files_and_labels:
    ysig = load_wav(fp)
    ybp  = bandpass_1_5k(ysig)
    feat = features25(ybp, SR).reshape(1, -1)
    Z = pca.transform(scaler.transform(feat))
    score = float(gmm.score_samples(Z)[0])   # log-likelihood; lower = more anomalous
    y_pred = int(score < thr)
    rows.append({
        "filepath": str(fp),
        "group_id": group_id_from_fp(fp),
        "y_true": int(y),
        "score": score,
        "y_pred": y_pred
    })

seg_df = pd.DataFrame(rows).sort_values("filepath")
# segment-level metrics (sanity)
m_seg = metrics_of(seg_df["y_true"].values, seg_df["y_pred"].values)

# ---- file-level aggregation ----
file_rows = []
for gid, grp in seg_df.groupby("group_id"):
    y_true_file = int(round(grp["y_true"].mean()))  # they should all match; fallback = majority
    mean_score  = float(grp["score"].mean())
    med_score   = float(grp["score"].median())
    frac_anom   = float((grp["y_pred"]==1).mean())
    # decision by MEAN score vs thr (consistent with training direction)
    y_pred_file = int(mean_score < thr)
    file_rows.append({
        "group_id": gid,
        "n_segments": int(len(grp)),
        "y_true_file": y_true_file,
        "mean_score": mean_score,
        "median_score": med_score,
        "frac_segments_pred_anom": frac_anom,
        "y_pred_file": y_pred_file
    })

file_df = pd.DataFrame(file_rows).sort_values("group_id")
m_file = metrics_of(file_df["y_true_file"].values, file_df["y_pred_file"].values)

# ---- outputs ----
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_seg_csv  = ART / f"segments_A8_GMM_{stamp}.csv"
out_file_csv = ART / f"files_A8_GMM_{stamp}.csv"
seg_df.to_csv(out_seg_csv, index=False)
file_df.to_csv(out_file_csv, index=False)

# confusion matrix (file-level)
cm_path = save_cm(m_file["cm"], "A8_GMM_FILE", ART)

# ---- small report ----
report_text = f"""
Per-file aggregation report — A8-GMM
Timestamp: {stamp}

What & why:
- We aggregate decisions per original recording (grouped by prefix before "__seg####").
- This matches deployment (alarm per file), reduces segment noise, and gives fairer, more stable metrics.

How:
- We re-computed A8 features on TEST: BP 1–5 kHz -> 25D features -> StandardScaler -> PCA(4) -> GMM.
- We used the saved validation-optimized threshold (thr = {thr:.6f}) on GMM log-likelihood scores.
- Segment decision: score < thr -> anomaly.
- File decision: mean(score) across segments < thr -> anomaly (we also log median & fraction anomalous).

Why the results matter:
- Segment-level metrics show raw detector sensitivity.
- File-level metrics reflect actual decision units in practice.
- If file-level F1 stays strong (or improves), the variant is robust for real-world use.

Results (TEST):
- Segment-level: acc={m_seg['acc']:.3f}, prec={m_seg['precision']:.3f}, rec={m_seg['recall']:.3f}, f1={m_seg['f1']:.3f}
- File-level:   acc={m_file['acc']:.3f}, prec={m_file['precision']:.3f}, rec={m_file['recall']:.3f}, f1={m_file['f1']:.3f}

Artifacts:
- Segment CSV: {out_seg_csv.name}
- File CSV:    {out_file_csv.name}
- CM (file):   {Path(cm_path).name}

Interpretation guide:
- If file-level F1 > segment-level F1: aggregation reduced false alarms (good).
- If recall drops a little but precision rises a lot: aggregation is filtering noise (likely good trade-off).
- Inspect files_A8_GMM_*.csv for borderline cases: mean_score close to thr.
""".strip()
report_path = ART / f"report_A8_GMM_FILE_{stamp}.txt"
report_path.write_text(report_text, encoding="utf-8")

# ---- log row + bundle run folder ----
params = dict(
    base_run="A8-GMM",
    mode="aggregation(mean score vs thr)",
    sr=SR, win_s=WIN_S, preprocess=["bandpass_1_5k"], features="25D",
    scaler="zscore", pca_dims=4, model="GMM(diag,3)", thr=thr, rng=RNG
)
code_txt = Path(__file__).read_text(encoding="utf-8")
rd = log_all(
    RUN_ID, CHANGE, MODEL_NAME,
    {"acc":m_file["acc"],"precision":m_file["precision"],"recall":m_file["recall"],"f1":m_file["f1"]},
    params, Path(cm_path), code_txt,
    extras={"seg_csv":out_seg_csv, "file_csv":out_file_csv, "report":report_path}
)

print("\n=== A35a (A8-GMM per-file) ===")
print("Segment-level:", {k: round(m_seg[k],3) for k in ["acc","precision","recall","f1"]})
print("File-level:   ", {k: round(m_file[k],3) for k in ["acc","precision","recall","f1"]})
print(f"Saved: {rd}")
