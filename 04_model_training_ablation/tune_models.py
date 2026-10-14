from pathlib import Path
import json, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT=Path(__file__).resolve().parent; MAN=ROOT/"manifests"; FEATS=["rms","zcr","centroid_hz","bandwidth_hz","flatness","kurtosis","crest","bp_0_1k","bp_1_3k","bp_3_8k","hf_ratio"]
def load(name):
    df=pd.read_csv(MAN/f"features_{name}.csv")
    y=(df["class"]=="anomaly").astype(int).values
    X=df[FEATS].values.astype(np.float32); return X,y
Xtr,ytr=load("train"); Xva,yva=load("val"); Xte,yte=load("test")
sc=StandardScaler().fit(Xtr); Xtr=sc.transform(Xtr); Xva=sc.transform(Xva); Xte=sc.transform(Xte)

def m(y,yp): 
    return dict(acc=accuracy_score(y,yp),prec=precision_score(y,yp,zero_division=0),
                rec=recall_score(y,yp),f1=f1_score(y,yp))

# GMM healthy-only with n_components sweep
best_g=(0,None,None)
for k in [1,2,3]:
    g=GaussianMixture(n_components=k,covariance_type="diag",random_state=0).fit(Xtr[ytr==0])
    s=g.score_samples(Xva)
    ths=np.percentile(s,np.linspace(1,99,99))
    for th in ths:
        ypv=(s<th).astype(int)
        f1=f1_score(yva,ypv)
        if f1>best_g[0]: best_g=(f1,g,th)
f1v,g,th=best_g
ypt=(g.score_samples(Xte)<th).astype(int); gm=m(yte,ypt)

# MLP small grid
best_m=(0,None)
for hid in [(16,),(32,),(32,16)]:
    for a in [1e-4,1e-3]:
        clf=MLPClassifier(hidden_layer_sizes=hid,alpha=a,max_iter=1000,early_stopping=True,random_state=0)
        w=(ytr==0).sum()/(ytr==1).sum(); sw=np.where(ytr==1,w,1.0)
        clf.fit(Xtr,ytr,sample_weight=sw)
        yp=clf.predict(Xva); f1=f1_score(yva,yp)
        if f1>best_m[0]: best_m=(f1,(hid,a,clf))
f1v,(hid,a,clf)=best_m
ml=m(yte,clf.predict(Xte))

print("== Tuned results (test) ==")
print(f"GMM diag k∈[1,3]: acc={gm['acc']:.3f} P={gm['prec']:.3f} R={gm['rec']:.3f} F1={gm['f1']:.3f}")
print(f"MLP {hid} alpha={a}: acc={ml['acc']:.3f} P={ml['prec']:.3f} R={ml['rec']:.3f} F1={ml['f1']:.3f}")
