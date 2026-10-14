# A35c: Per-file aggregation for A31-MLP (SR=16k, BP1–5k, Hilbert env + 20ms smooth + log norm95,
#       25-D features -> MinMax -> PCA(8) -> MLP 1x10)
# - Loads artifacts/mlp_A31.joblib
# - Scores TEST segments, groups by "<prefix>__seg####.wav"
# - Writes segment/file CSVs, file-level CM, short report, and logs a row to variants.csv
# - Robust logging: if variants.csv is locked (e.g., open in Excel), appends to variants_queue.csv instead.

from pathlib import Path
from datetime import datetime
import json, shutil, csv
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
TEST = HERE / "data" / "segments_split" / "test"
ART  = HERE / "artifacts"
RUNS = ART / "runs"
ART.mkdir(parents=True, exist_ok=True); RUNS.mkdir(parents=True, exist_ok=True)
print(f"[PATH] TEST = {TEST}"); print(f"[PATH] ART  = {ART}")

# ---- constants (match A31) ----
SR=16000; WIN_S=2.0; BP_LOW=1000; BP_HIGH=5000; SMOOTH_MS=20; NORM_PCT=95; RNG=42
MODEL_PATH = ART/"mlp_A31.joblib"
RUN_ID="A31-MLP-FILE"; CHANGE="Per-file aggregation of A31 (env+log@16k) by mean prob>=0.5"; MODEL_NAME="MLP-file"

# ---- helpers ----
def load_wav(fp):
    y, sr = sf.read(fp, dtype='float32', always_2d=False)
    if y.ndim>1: y=y[:,0]
    if sr!=SR: y=librosa.resample(y, orig_sr=sr, target_sr=SR)
    return y

def moving_average(x, k):
    if k<=1: return x.astype(np.float32)
    k=int(k); return np.convolve(x.astype(np.float32), np.ones(k, np.float32)/k, mode='same')

def bp_hilbert_env_log(y, sr=SR, f1=BP_LOW, f2=BP_HIGH, smooth_ms=SMOOTH_MS, pct=NORM_PCT):
    sos = sig.iirfilter(N=4, Wn=(f1, f2), btype='bandpass', fs=sr, output='sos')
    y_bp = sig.sosfiltfilt(sos, y).astype(np.float32)
    env = np.abs(sig.hilbert(y_bp)).astype(np.float32)
    win = max(1, int(round(smooth_ms*sr/1000.0)))
    env_s = moving_average(env, win)
    scale = np.percentile(env_s, pct) + 1e-8
    env_ln = np.log1p(env_s/scale).astype(np.float32)
    rms_o=np.sqrt(np.mean(y**2)+1e-12); rms_n=np.sqrt(np.mean(env_ln**2)+1e-12)
    if rms_n>0: env_ln*= (rms_o/rms_n)
    return env_ln

def bandpower(y, sr, fmin, fmax):
    S = np.abs(np.fft.rfft(y))**2; freqs = np.fft.rfftfreq(len(y), 1/sr)
    idx=(freqs>=fmin)&(freqs<fmax)
    return float(S[idx].sum()/(S.sum()+1e-12))

def spectral_slope(y):
    S=np.abs(np.fft.rfft(y)); x=np.arange(len(S)); xm,ym=x.mean(),S.mean()
    return float(((x-xm)*(S-ym)).sum()/(((x-xm)**2).sum()+1e-12))

def autocorr_feats(y, max_lag=200):
    y=y-y.mean(); ac=np.correlate(y,y,mode='full')[len(y)-1:len(y)+max_lag]
    if len(ac)<2: return 0.0,0
    ac[0]=0.0; k=int(np.argmax(ac)); return float(ac[k]/(np.sum(y**2)+1e-12)), int(k)

def features25(y, sr):
    if len(y)<int(SR*WIN_S): y=np.pad(y,(0,int(SR*WIN_S)-len(y)))
    x=y.astype(np.float32); rms=float(np.sqrt(np.mean(x**2)+1e-12))
    feat=[float(np.mean(x)), float(np.std(x)), rms,
          float(np.max(np.abs(x))), float(np.ptp(x)),
          float(skew(x)), float(kurtosis(x,fisher=True)),
          float(np.mean(np.abs(np.diff(x)))), float(np.mean(np.square(np.diff(x)))),
          float(np.count_nonzero(librosa.zero_crossings(x)))/len(x)]
    bands=[(0,250),(250,500),(500,1000),(1000,2000),(2000,3000),(3000,4000),(4000,6000),(6000,8000)]
    for f1,f2 in bands: feat.append(bandpower(x, sr, f1, f2))
    S=np.abs(np.fft.rfft(x)); freqs=np.fft.rfftfreq(len(x),1/sr)
    centroid=float((freqs*(S+1e-12)).sum()/(S.sum()+1e-12))
    bw=float(np.sqrt(((freqs-centroid)**2*(S+1e-12)).sum()/(S.sum()+1e-12)))
    roll=float(librosa.feature.spectral_rolloff(y=x, sr=sr, roll_percent=0.85).mean())
    flat=float(librosa.feature.spectral_flatness(y=x).mean())
    slope=spectral_slope(x); ac_val,ac_lag=autocorr_feats(x)
    feat+=[centroid,bw,roll,flat,slope,ac_val,float(ac_lag)]
    fv=np.asarray(feat,np.float32)
    if fv.shape[0]!=25: raise RuntimeError(f"features25={fv.shape[0]}!=25")
    return fv

def group_id(fp: Path)->str: return fp.stem.split("__seg")[0]

def metrics_of(y_true,y_pred):
    return dict(acc=accuracy_score(y_true,y_pred),
                precision=precision_score(y_true,y_pred,zero_division=0),
                recall=recall_score(y_true,y_pred,zero_division=0),
                f1=f1_score(y_true,y_pred,zero_division=0),
                cm=confusion_matrix(y_true,y_pred))

def save_cm(cm, name, outdir: Path):
    fig=plt.figure(); plt.imshow(cm, interpolation='nearest'); plt.title(name)
    plt.xlabel('Pred'); plt.ylabel('True')
    for (i,j),v in np.ndenumerate(cm): plt.text(j,i,str(v),ha='center',va='center')
    out=outdir/f"cm_{name}.png"; plt.tight_layout(); fig.savefig(out,dpi=160); plt.close(fig); return out

def log_all(run_id, change, model, metrics_row, params, cm_path, code_text, extras: dict):
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    rd=RUNS/f"{run_id}_{stamp}"; rd.mkdir(parents=True, exist_ok=True)
    (rd/"params.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    mfull=dict(metrics_row); mfull["timestamp"]=stamp
    (rd/"metrics.json").write_text(json.dumps(mfull, indent=2, ensure_ascii=False), encoding="utf-8")
    (rd/"code.py").write_text(code_text, encoding="utf-8")
    cm_local=""
    if cm_path and Path(cm_path).exists():
        cm_local=str(rd/Path(cm_path).name); shutil.copy2(cm_path, cm_local)
    for _,p in extras.items():
        if isinstance(p,Path) and p.exists(): shutil.copy2(p, rd/p.name)

    row={"timestamp":stamp,"id":run_id,"change":change,"model":model,
         "acc":metrics_row.get("acc"),"precision":metrics_row.get("precision"),
         "recall":metrics_row.get("recall"),"f1":metrics_row.get("f1"),
         "run_dir":str(rd),"code_path":str(rd/'code.py'),
         "cm_path":cm_local or str(cm_path or ""), "params_path":str(rd/'params.json'),
         "metrics_path":str(rd/'metrics.json'),"model_path":"","notes":"Per-file aggregation"}

    csv_path=ART/"variants.csv"; write_header=not csv_path.exists()
    try:
        with open(csv_path,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header: w.writeheader()
            w.writerow(row)
        print(f"[OK] appended: {csv_path}")
    except PermissionError:
        queue=ART/"variants_queue.csv"; q_header=not queue.exists()
        with open(queue,"a",newline="",encoding="utf-8") as f:
            w=csv.DictWriter(f, fieldnames=list(row.keys()))
            if q_header: w.writeheader()
            w.writerow(row)
        print(f"[WARN] variants.csv locked (likely open). Appended to queue: {queue}")

    print(f"[OK] saved: {rd}")
    return rd

# ---- load model ----
if not MODEL_PATH.exists(): raise FileNotFoundError(f"Missing {MODEL_PATH} (run A31 first)")
bundle=joblib.load(MODEL_PATH); scaler=bundle["scaler"]; pca=bundle["pca"]; mlp=bundle["mlp"]

# ---- collect TEST files ----
normals=sorted((TEST/"normal").glob("*.wav")); anomalies=sorted((TEST/"anomaly").glob("*.wav"))
if not normals and not anomalies: raise FileNotFoundError(f"No WAVs under {TEST}/(normal|anomaly)")
files=[(fp,0) for fp in normals]+[(fp,1) for fp in anomalies]

# ---- segment-level ----
rows=[]
for fp,y in files:
    sig_in=load_wav(fp)
    sig_p = bp_hilbert_env_log(sig_in)
    feat  = features25(sig_p, SR).reshape(1,-1)
    Z     = pca.transform(scaler.transform(feat))
    proba = float(mlp.predict_proba(Z)[0,1])
    y_pred= int(proba>=0.5)
    rows.append({"filepath":str(fp),"group_id":group_id(fp),"y_true":int(y),
                 "proba_anom":proba,"y_pred":y_pred})
seg_df=pd.DataFrame(rows).sort_values("filepath")
m_seg=metrics_of(seg_df.y_true.values, seg_df.y_pred.values)

# ---- file-level (mean prob) ----
file_rows=[]
for gid,grp in seg_df.groupby("group_id"):
    y_true=int(round(grp.y_true.mean()))
    mean_p=float(grp.proba_anom.mean()); med_p=float(grp.proba_anom.median())
    frac=float((grp.y_pred==1).mean())
    y_pred=int(mean_p>=0.5)
    file_rows.append({"group_id":gid,"n_segments":int(len(grp)),
                      "y_true_file":y_true,"mean_proba_anom":mean_p,
                      "median_proba_anom":med_p,"frac_segments_pred_anom":frac,
                      "y_pred_file":y_pred})
file_df=pd.DataFrame(file_rows).sort_values("group_id")
m_file=metrics_of(file_df.y_true_file.values, file_df.y_pred_file.values)

# ---- outputs ----
stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
out_seg=ART/f"segments_A31_MLP_{stamp}.csv"
out_file=ART/f"files_A31_MLP_{stamp}.csv"
seg_df.to_csv(out_seg,index=False); file_df.to_csv(out_file,index=False)
cm_path=save_cm(m_file["cm"], "A31_MLP_FILE", ART)

report=f"""
Per-file aggregation report — A31-MLP
Timestamp: {stamp}

What & why:
- Aggregate segment predictions per recording (prefix before "__seg####") to match deployment and reduce noise.

How:
- TEST pipeline (A31): SR16k -> BP 1–5 kHz -> Hilbert envelope -> {SMOOTH_MS} ms moving average -> log1p with {NORM_PCT}th-percentile norm.
- 25-D features -> MinMax scaler -> PCA(8) -> MLP(1x10, ReLU).
- Segment decision: prob(anom) >= 0.5; File decision: mean prob >= 0.5.

Why results matter:
- Segment metrics = raw detector behavior; File metrics = real decision units.
- If file-level precision improves with similar recall, aggregation is beneficial.

Results (TEST):
- Segment-level: acc={m_seg['acc']:.3f}, prec={m_seg['precision']:.3f}, rec={m_seg['recall']:.3f}, f1={m_seg['f1']:.3f}
- File-level:   acc={m_file['acc']:.3f}, prec={m_file['precision']:.3f}, rec={m_file['recall']:.3f}, f1={m_file['f1']:.3f}

Artifacts:
- {out_seg.name}
- {out_file.name}
- {Path(cm_path).name}
""".strip()
rep_path=ART/f"report_A31_MLP_FILE_{stamp}.txt"
rep_path.write_text(report, encoding="utf-8")

params=dict(base_run="A31-MLP", mode="aggregation(mean probability >= 0.5)",
            sr=SR, win_s=WIN_S, preprocess=[f"bandpass_{BP_LOW}_{BP_HIGH}","hilbert_envelope",
            f"smooth_{SMOOTH_MS}ms", f"log1p_norm{NORM_PCT}pct"],
            features="25D", scaler="minmax", pca_dims=8, model="MLP(1x10, ReLU)", rng=RNG)
code_txt=Path(__file__).read_text(encoding="utf-8")
rd=log_all(RUN_ID, CHANGE, MODEL_NAME,
           {"acc":m_file["acc"],"precision":m_file["precision"],"recall":m_file["recall"],"f1":m_file["f1"]},
           params, Path(cm_path), code_txt, extras={"seg_csv":out_seg,"file_csv":out_file,"report":rep_path})

print("\n=== A35c (A31-MLP per-file) ===")
print("Segment-level:", {k:round(m_seg[k],3) for k in ["acc","precision","recall","f1"]})
print("File-level:   ", {k:round(m_file[k],3) for k in ["acc","precision","recall","f1"]})
print(f"Saved: {rd}")
