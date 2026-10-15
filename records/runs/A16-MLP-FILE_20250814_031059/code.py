# A35b: Per-file aggregation for A16-MLP (env+log @ 8 kHz, PCA(8), MinMax, MLP 1x10)
# - Loads saved model: artifacts/mlp_A16.joblib
# - Re-scores TEST segments, groups by "<prefix>__seg####.wav" prefix
# - Segment-level CSV + metrics
# - File-level CSV + metrics + confusion matrix
# - Writes a short report.txt (what/why/how + why results matter)
# - Appends a row to artifacts/variants.csv and bundles artifacts under artifacts/runs/

from pathlib import Path
from datetime import datetime
import json, shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import librosa, scipy.signal as sig
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.stats import skew, kurtosis

# ---- fixed paths ----
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "segments_split" / "test"
ART  = HERE / "artifacts"
RUNS = ART / "runs"
ART.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
print(f"[PATH] TEST = {DATA}"); print(f"[PATH] ART  = {ART}")

# ---- constants (match A16 family) ----
SR = 8000
WIN_S = 2.0
SMOOTH_MS = 20
NORM_PCT = 95
BP_LOW = 1000
BP_HIGH_DESIRED = 5000  # will be clamped to < Nyquist for SR=8k
RNG = 42

MODEL_PATH = ART / "mlp_A16.joblib"   # must exist (from your A16 run)
RUN_ID = "A16-MLP-FILE"
CHANGE = "Per-file aggregation of A16 (env+log @8k) with mean prob >= 0.5"
MODEL_NAME = "MLP-file"

# ---- helpers ----
def load_wav(fp):
    y, sr = sf.read(fp, dtype='float32', always_2d=False)
    if y.ndim>1: y = y[:,0]
    if sr != SR: y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y

def moving_average(x, k):
    if k <= 1: return x
    k = int(k)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(x.astype(np.float32), kernel, mode='same')

def bp_hilbert_envelope_log(y, sr=SR, f1=BP_LOW, f2_desired=BP_HIGH_DESIRED, smooth_ms=SMOOTH_MS, pct=NORM_PCT):
    # Clamp high cutoff below Nyquist for stability (SR=8k -> ~3.6 kHz with 0.45*sr)
    f2 = min(f2_desired, 0.45*sr)
    if f1 >= f2:  # fallback: no BP, just envelope
        y_f = y.astype(np.float32)
    else:
        sos = sig.iirfilter(N=4, Wn=(f1, f2), btype='bandpass', fs=sr, output='sos')
        y_f = sig.sosfiltfilt(sos, y).astype(np.float32)
    env = np.abs(sig.hilbert(y_f)).astype(np.float32)
    win = max(1, int(round(smooth_ms * sr / 1000.0)))
    env_s = moving_average(env, win)
    scale = np.percentile(env_s, pct) + 1e-8
    env_ln = np.log1p(env_s / scale).astype(np.float32)
    # RMS match (keeps 25D features scale stable across files)
    rms_o = np.sqrt(np.mean(y**2)+1e-12); rms_n = np.sqrt(np.mean(env_ln**2)+1e-12)
    if rms_n > 0: env_ln *= (rms_o / rms_n)
    return env_ln

def bandpower(y, sr, fmin, fmax):
    S = np.abs(np.fft.rfft(y))**2
    freqs = np.fft.rfftfreq(len(y), 1/sr)
    idx = (freqs>=fmin)&(freqs<fmax)
    # If idx is empty (e.g., band above Nyquist), sum is 0 -> safe
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
    """Return EXACTLY 25 features (match training)."""
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
    # Keep the SAME 8 bands as 16 kHz setup (last two will be ~0 at 8 kHz)
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
    fv = np.asarray(feat, dtype=np.float32)
    # Safety check
    if fv.shape[0] != 25:
        raise RuntimeError(f"features25 produced {fv.shape[0]} dims, expected 25")
    return fv

def group_id_from_fp(fp: Path) -> str:
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
    (rd/"params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    mfull = dict(metrics_row); mfull["timestamp"]=stamp
    (rd/"metrics.json").write_text(json.dumps(mfull, indent=2, ensure_ascii=False), encoding="utf-8")
    (rd/"code.py").write_text(code_text, encoding="utf-8")
    cm_local=""
    if cm_path and Path(cm_path).exists():
        cm_local=str(rd/Path(cm_path).name); shutil.copy2(cm_path, cm_local)
    for k, p in extras.items():
        if isinstance(p, Path) and p.exists():
            shutil.copy2(p, rd/p.name)
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
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH} (run A16 first)")
bundle = joblib.load(MODEL_PATH)
scaler = bundle["scaler"]; pca = bundle["pca"]; mlp = bundle["mlp"]

# ---- collect TEST files ----
normal_dir  = DATA / "normal"
anomaly_dir = DATA / "anomaly"
normals  = sorted(normal_dir.glob("*.wav"))
anomalies= sorted(anomaly_dir.glob("*.wav"))
if not normals and not anomalies:
    raise FileNotFoundError(f"No WAVs under {DATA}/(normal|anomaly)")
files_and_labels = [(fp,0) for fp in normals] + [(fp,1) for fp in anomalies]

# ---- segment-level predictions ----
rows = []
for fp, y in files_and_labels:
    ysig = load_wav(fp)
    yproc = bp_hilbert_envelope_log(ysig)
    feat = features25(yproc, SR).reshape(1, -1)
    Z = pca.transform(scaler.transform(feat))
    proba = float(mlp.predict_proba(Z)[0,1])
    y_pred = int(proba >= 0.5)
    rows.append({
        "filepath": str(fp),
        "group_id": group_id_from_fp(fp),
        "y_true": int(y),
        "proba_anom": proba,
        "y_pred": y_pred
    })

seg_df = pd.DataFrame(rows).sort_values("filepath")
m_seg = metrics_of(seg_df["y_true"].values, seg_df["y_pred"].values)

# ---- file-level aggregation ----
file_rows = []
for gid, grp in seg_df.groupby("group_id"):
    y_true_file = int(round(grp["y_true"].mean()))
    mean_p  = float(grp["proba_anom"].mean())
    med_p   = float(grp["proba_anom"].median())
    frac_anom = float((grp["y_pred"]==1).mean())
    y_pred_file = int(mean_p >= 0.5)  # decision by mean probability
    file_rows.append({
        "group_id": gid,
        "n_segments": int(len(grp)),
        "y_true_file": y_true_file,
        "mean_proba_anom": mean_p,
        "median_proba_anom": med_p,
        "frac_segments_pred_anom": frac_anom,
        "y_pred_file": y_pred_file
    })

file_df = pd.DataFrame(file_rows).sort_values("group_id")
m_file = metrics_of(file_df["y_true_file"].values, file_df["y_pred_file"].values)

# ---- outputs ----
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_seg_csv  = ART / f"segments_A16_MLP_{stamp}.csv"
out_file_csv = ART / f"files_A16_MLP_{stamp}.csv"
seg_df.to_csv(out_seg_csv, index=False)
file_df.to_csv(out_file_csv, index=False)
cm_path = save_cm(m_file["cm"], "A16_MLP_FILE", ART)

# ---- small report ----
report_text = f"""
Per-file aggregation report — A16-MLP
Timestamp: {stamp}

What & why:
- We aggregate segment predictions per original recording (prefix before "__seg####").
- File-level evaluation matches deployment, reduces segment noise, and yields fairer metrics.

How:
- TEST pipeline re-used: 8 kHz -> Hilbert envelope -> {SMOOTH_MS} ms moving average -> log1p with {NORM_PCT}th-percentile normalization.
- 25-D features -> MinMax scaler -> PCA(8) -> MLP(1x10, ReLU).
- Segment decision: prob(anomaly) >= 0.5.
- File decision: mean prob(anomaly) >= 0.5 (we also log median & fraction anomalous).

Why the results matter:
- Segment metrics show raw detector behavior; file metrics reflect actual user-facing decisions.
- If file-level F1 stays high or improves, the variant is robust and less sensitive to transient noise.

Results (TEST):
- Segment-level: acc={m_seg['acc']:.3f}, prec={m_seg['precision']:.3f}, rec={m_seg['recall']:.3f}, f1={m_seg['f1']:.3f}
- File-level:   acc={m_file['acc']:.3f}, prec={m_file['precision']:.3f}, rec={m_file['recall']:.3f}, f1={m_file['f1']:.3f}

Artifacts:
- Segment CSV: {out_seg_csv.name}
- File CSV:    {out_file_csv.name}
- CM (file):   {Path(cm_path).name}

Interpretation guide:
- If precision rises at file-level with similar recall, aggregation reduced false alarms (good).
- Mean_proba near 0.5 indicates borderline files worth manual listening/inspection.
""".strip()
report_path = ART / f"report_A16_MLP_FILE_{stamp}.txt"
report_path.write_text(report_text, encoding="utf-8")

# ---- log row + bundle run folder ----
params = dict(
    base_run="A16-MLP",
    mode="aggregation(mean probability >= 0.5)",
    sr=SR, win_s=WIN_S,
    preprocess=["hilbert_envelope", f"smooth_{SMOOTH_MS}ms", f"log1p_norm{NORM_PCT}pct", "bp_clamped"],
    features="25D",
    scaler="minmax", pca_dims=8, model="MLP(1x10, ReLU)", rng=RNG
)
code_txt = Path(__file__).read_text(encoding="utf-8")
rd = log_all(
    RUN_ID, CHANGE, MODEL_NAME,
    {"acc":m_file["acc"],"precision":m_file["precision"],"recall":m_file["recall"],"f1":m_file["f1"]},
    params, Path(cm_path), code_txt,
    extras={"seg_csv":out_seg_csv, "file_csv":out_file_csv, "report":report_path}
)

print("\n=== A35b (A16-MLP per-file) ===")
print("Segment-level:", {k: round(m_seg[k],3) for k in ["acc","precision","recall","f1"]})
print("File-level:   ", {k: round(m_file[k],3) for k in ["acc","precision","recall","f1"]})
print(f"Saved: {rd}")
