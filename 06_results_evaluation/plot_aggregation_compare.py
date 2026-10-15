# Aggregation rule compare (file-F1): mean (A35), majority (A40), median (A41) for A8 & A16
from pathlib import Path
import csv, numpy as np, matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ART=HERE/"artifacts"; FIGS=ART/"figs"; FIGS.mkdir(parents=True, exist_ok=True)
vc=ART/"variants.csv"

def last_row_id(id_exact):
    if not vc.exists(): return None
    with open(vc,"r",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    last=None
    for r in rows:
        rid=(r.get("id") or "").upper()
        if rid==id_exact.upper(): last=r
    return last

def f1(r):
    if not r: return None
    for k in ["file_F1","file_f1","f1"]:
        if k in r and r[k]!="":
            try: return float(r[k])
            except: pass
    return None

# mean = A8-GMM-FILE / A16-MLP-FILE; majority = A40-A*; median = A41-A*
A8_mean = last_row_id("A8-GMM-FILE")
A8_maj  = last_row_id("A40-A8")
A8_med  = last_row_id("A41-A8")
A16_mean= last_row_id("A16-MLP-FILE")
A16_maj = last_row_id("A40-A16")
A16_med = last_row_id("A41-A16")

rules=["mean","majority","median"]
vals8 =[f1(A8_mean), f1(A8_maj), f1(A8_med)]
vals16=[f1(A16_mean), f1(A16_maj), f1(A16_med)]

fig=plt.figure(figsize=(6,3.2), dpi=120); ax=fig.add_subplot(111)
x=np.arange(len(rules)); w=0.35
ax.bar(x-w/2, [v if v is not None else 0 for v in vals8], width=w, label="A8-GMM")
ax.bar(x+w/2, [v if v is not None else 0 for v in vals16], width=w, label="A16-MLP")
ax.set_xticks(x); ax.set_xticklabels(rules); ax.set_ylim(0,1); ax.set_ylabel("File F1")
for i,v in enumerate(vals8):
    if v is not None: ax.text(i-w/2, v+0.02, f"{v:.3f}", ha="center")
for i,v in enumerate(vals16):
    if v is not None: ax.text(i+w/2, v+0.02, f"{v:.3f}", ha="center")
ax.legend(); fig.tight_layout()
out=FIGS/"aggregation_compare.png"; fig.savefig(out); plt.close(fig)
print(f"[OK] saved: {out}")
