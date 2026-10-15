# B0 vs B1 vs B2 (file-F1) for A8-GMM and A16-MLP
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

A8 = {"B0": last_row_id("A8-GMM-FILE"),
      "B1": last_row_id("B1-A8"),
      "B2": last_row_id("B2-A8")}
A16 = {"B0": last_row_id("A16-MLP-FILE"),
       "B1": last_row_id("B1-A16"),
       "B2": last_row_id("B2-A16")}

labels=["B0","B1","B2"]
vals8=[f1(A8[k]) for k in labels]
vals16=[f1(A16[k]) for k in labels]

fig=plt.figure(figsize=(6,3.2), dpi=120); ax=fig.add_subplot(111)
x=np.arange(len(labels)); w=0.35
ax.bar(x-w/2, [v if v is not None else 0 for v in vals8], width=w, label="A8-GMM")
ax.bar(x+w/2, [v if v is not None else 0 for v in vals16], width=w, label="A16-MLP")
ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0,1); ax.set_ylabel("File F1")
for i,v in enumerate(vals8):
    if v is not None: ax.text(i-w/2, v+0.02, f"{v:.3f}", ha="center")
for i,v in enumerate(vals16):
    if v is not None: ax.text(i+w/2, v+0.02, f"{v:.3f}", ha="center")
ax.legend(); fig.tight_layout()
out=FIGS/"b0_b1_b2_compare.png"; fig.savefig(out); plt.close(fig)
print(f"[OK] saved: {out}")
