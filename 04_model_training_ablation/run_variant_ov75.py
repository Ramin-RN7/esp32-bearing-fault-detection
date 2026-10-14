from pathlib import Path
import numpy as np, csv, json, math, shutil
import soundfile as sf
from scipy.signal import resample_poly
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

# ----- VARIANT SETTINGS -----
WIN_SEC = 2.0
HOP_SEC = 0.5          # 75% overlap
SR = 16000
VAR = f"ov75_win{WIN_SEC:.1f}_hop{HOP_SEC:.1f}"
ROOT = Path(__file__).resolve().parent
RAW = ROOT/"data"/"raw"
WAV16K = ROOT/"data"/"wav16k"                   # reuse (safe)
SEGS = ROOT/"data"/f"segments_{VAR}"
MAN = ROOT/f"manifests_{VAR}"
SPLITDIR = ROOT/"data"/f"segments_split_{VAR}"
ART = ROOT/"artifacts"/VAR; ART.mkdir(parents=True, exist_ok=True)
CLASSES = ["normal","anomaly"]

# ----- UTILS -----
def to_mono(y): return y.mean(axis=1) if y.ndim==2 else y
def norm_peak(y, target_dbfs=-1.0):
    t = 10**(target_dbfs/20.0); peak = float(np.max(np.abs(y))+1e-12)
    return y * min(1.0, t/peak)
def standardize(src:Path, dst:Path):
    y,sr = sf.read(src, always_2d=False)
    y = to_mono(np.asarray(y, np.float32))
    if sr!=SR: y = resample_poly(y, up=SR, down=sr).astype(np.float32)
    y = np.clip(norm_peak(y), -1.0, 1.0)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, (y*32767).astype(np.int16), SR, subtype="PCM_16")
def segment(src:Path, outdir:Path, orig:str, cls:str, writer):
    y,sr = sf.read(src, dtype="int16"); assert sr==SR
    y = y.astype(np.float32)/32767.0
    win, hop = int(WIN_SEC*SR), int(HOP_SEC*SR)
    nsegs=0
    for start in range(0, len(y)-win+1, hop):
        seg=y[start:start+win]; nsegs+=1
        name=f"{Path(orig).stem}__seg{nsegs:04d}.wav"
        out=(outdir/cls/name); out.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out, (np.clip(seg,-1,1)*32767).astype(np.int16), SR, subtype="PCM_16")
        writer.writerow({"filepath":str(out.relative_to(ROOT)),"class":cls,"orig_file":orig,
                         "segment_id":f"{nsegs:04d}","duration_s":f"{WIN_SEC:.2f}"})
    return nsegs

# ----- 1) STANDARDIZE (skip if exists) -----
for cls in CLASSES:
    for p in sorted((RAW/cls).glob("*.wav")):
        out = WAV16K/cls/p.name
        if not out.exists(): standardize(p, out)

# ----- 2) SEGMENT -----
MAN.mkdir(parents=True, exist_ok=True)
with open(MAN/"segments.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=["filepath","class","orig_file","segment_id","duration_s"]); w.writeheader()
    sums={c:0 for c in CLASSES}
    for cls in CLASSES:
        for p in tqdm(sorted((WAV16K/cls).glob("*.wav")), desc=f"seg:{cls}"):
            sums[cls]+=segment(p, SEGS, p.name, cls, w)
print("Segments:", sums)

# ----- 3) SPLIT (by original file) -----
import random
IN = list(csv.DictReader(open(MAN/"segments.csv")))
byfile={c:{} for c in CLASSES}
for r in IN: byfile[r["class"]].setdefault(r["orig_file"],[]).append(r)
assign={}
for c in CLASSES:
    files=list(byfile[c].keys()); rnd=random.Random(42); rnd.shuffle(files)
    n=len(files); ntr=math.floor(0.70*n); nva=max(1, round(0.15*n)); nte=n-ntr-nva
    for of in files[:ntr]: assign[of]="train"
    for of in files[ntr:ntr+nva]: assign[of]="val"
    for of in files[ntr+nva:]: assign[of]="test"

SPLITDIR.mkdir(parents=True, exist_ok=True)
for sp in ["train","val","test"]:
    for c in CLASSES: (SPLITDIR/sp/c).mkdir(parents=True, exist_ok=True)

def write_split(name):
    out=MAN/f"split_{name}.csv"
    with open(out,"w",newline="") as f:
        fields=["filepath","class","orig_file","segment_id","duration_s","split"]
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in IN:
            if assign[r["orig_file"]]==name:
                rr=dict(r); rr["split"]=name; w.writerow(rr)
                src=(ROOT/r["filepath"]).resolve()
                dst=(SPLITDIR/name/r["class"]/f"{Path(r['filepath']).stem}.wav").resolve()
                shutil.copy2(src,dst)
for sp in ["train","val","test"]: write_split(sp)

# ----- 4) FEATURES (same 11 as baseline) -----
def load_audio(p): y,sr=sf.read(p,dtype="int16"); assert sr==SR; return (y.astype(np.float32)/32767.0)
def features(y):
    rms=float(np.sqrt(np.mean(y**2)+1e-12))
    zcr=float(((y[:-1]*y[1:])<0).mean())
    mean=float(np.mean(y)); std=float(np.std(y)+1e-12)
    kurt=float(np.mean(((y-mean)/std)**4)); crest=float((np.max(np.abs(y))+1e-12)/(rms+1e-12))
    n=len(y); w=np.hanning(n); NFFT=4096
    Y=np.fft.rfft(y*w, n=NFFT); psd=(np.abs(Y)+1e-12)**2; freqs=np.fft.rfftfreq(NFFT, d=1.0/SR)
    s=float(np.sum(psd)+1e-20)
    centroid=float(np.sum(freqs*psd)/s); bw=float(np.sqrt(np.sum(((freqs-centroid)**2)*psd)/s))
    flat=float(np.exp(np.mean(np.log(psd)))/(np.mean(psd)+1e-12))
    def bp(lo,hi): idx=np.where((freqs>=lo)&(freqs<hi))[0]; return float(np.sum(psd[idx])/s)
    bp0=bp(0,1000); bp1=bp(1000,3000); bp2=bp(3000,8000); hfr=float(bp2/(bp0+bp1+1e-12))
    return [rms,zcr,centroid,bw,flat,kurt,crest,bp0,bp1,bp2,hfr]
FEATS=["rms","zcr","centroid_hz","bandwidth_hz","flatness","kurtosis","crest","bp_0_1k","bp_1_3k","bp_3_8k","hf_ratio"]
def make_feats(split):
    rows=list(csv.DictReader(open(MAN/f"split_{split}.csv")))
    with open(MAN/f"features_{split}.csv","w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["filepath","class","orig_file","segment_id","duration_s"]+FEATS); w.writeheader()
        for r in tqdm(rows, desc=f"feat:{split}"):
            y=load_audio(ROOT/r["filepath"]); vals=features(y)
            out={k:r[k] for k in ["filepath","class","orig_file","segment_id","duration_s"]}
            for k,v in zip(FEATS,vals): out[k]=v; w.writerow(out)
for sp in ["train","val","test"]: make_feats(sp)

# ----- 5) TRAIN + CALIBRATE -----
def load(split):
    import pandas as pd
    df=pd.read_csv(MAN/f"features_{split}.csv")
    y=(df["class"]=="anomaly").astype(int).values
    X=df[FEATS].values.astype(np.float32); return X,y
Xtr,ytr=load("train"); Xva,yva=load("val"); Xte,yte=load("test")
sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xva=sc.transform(Xva); Xte=sc.transform(Xte)

# GMM (healthy-only) with threshold from val
g=GaussianMixture(n_components=1,covariance_type="diag",random_state=0).fit(Xtr[ytr==0])
vs=g.score_samples(Xva); th=np.percentile(vs, 10)  # coarse; val-opt could be grid
ypg=(g.score_samples(Xte)<th).astype(int)

# MLP tiny
w=(ytr==0).sum()/(ytr==1).sum(); sw=np.where(ytr==1,w,1.0)
mlp=MLPClassifier(hidden_layer_sizes=(16,),alpha=1e-4,max_iter=1000,early_stopping=True,random_state=0).fit(Xtr,ytr,sample_weight=sw)
ypm=mlp.predict(Xte)

def metr(y,yp): 
    return dict(acc=accuracy_score(y,yp),prec=precision_score(y,yp,zero_division=0),
                rec=recall_score(y,yp),f1=f1_score(y,yp),cm=confusion_matrix(y,yp).tolist())

mg=metr(yte, ypg); mm=metr(yte, ypm)
print("== Variant:", VAR)
print(f"GMM: acc={mg['acc']:.3f} P={mg['prec']:.3f} R={mg['rec']:.3f} F1={mg['f1']:.3f}")
print(f"MLP: acc={mm['acc']:.3f} P={mm['prec']:.3f} R={mm['rec']:.3f} F1={mm['f1']:.3f}")

# Save CMs and append variants.csv
def save_cm(y,yp,name):
    cm=confusion_matrix(y,yp); fig=plt.figure(figsize=(3,3))
    plt.imshow(cm); plt.title(name); plt.xlabel("Pred"); plt.ylabel("True")
    for i in range(2):
        for j in range(2):
            plt.text(j,i,int(cm[i,j]),ha="center",va="center")
    plt.tight_layout(); out=ART/f"cm_{name}.png"; fig.savefig(out,dpi=200); plt.close(fig)
save_cm(yte, ypg, "gmm_test"); save_cm(yte, ypm, "mlp_test")

varcsv=ROOT/"artifacts"/"variants.csv"
hdr="variant,win_s,hop_s,features,model,acc,prec,rec,f1\n"
if not varcsv.exists(): varcsv.write_text(hdr)
with open(varcsv,"a") as f:
    f.write(f"{VAR},{WIN_SEC},{HOP_SEC},stat11,GMM,{mg['acc']:.4f},{mg['prec']:.4f},{mg['rec']:.4f},{mg['f1']:.4f}\n")
    f.write(f"{VAR},{WIN_SEC},{HOP_SEC},stat11,MLP,{mm['acc']:.4f},{mm['prec']:.4f},{mm['rec']:.4f},{mm['f1']:.4f}\n")
print("Wrote:", varcsv, "and artifacts in", ART)
