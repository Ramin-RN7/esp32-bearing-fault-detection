# Top-3 file-F1 bar: A8-GMM-FILE, A16-MLP-FILE, A31-MLP-FILE
from pathlib import Path
import csv, re
import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ART = HERE/"artifacts"; FIGS = ART/"figs"; FIGS.mkdir(parents=True, exist_ok=True)
vc = ART/"variants.csv"

def best_row_like(patterns):
    # return latest row matching ALL substrings in id (case-insensitive)
    best=None
    if not vc.exists(): return None
    with open(vc, "r", encoding="utf-8") as f:
        r=csv.DictReader(f); rows=list(r)
    for row in rows:
        rid = (row.get("id") or row.get("run_id") or "").lower()
        if all(p.lower() in rid for p in patterns):
            best=row
    return best

def get_f1(row):
    if not row: return None
    for k in ["file_F1","file_f1","f1"]:
        if k in row and str(row[k]).strip()!="":
            try: return float(row[k])
            except: pass
    return None

pairs = [
    (["a8","gmm","file"], "A8-GMM"),
    (["a16","mlp","file"],"A16-MLP"),
    (["a31","mlp","file"],"A31-MLP"),
]

labs, vals = [], []
for pats, lab in pairs:
    row = best_row_like(pats)
    f1 = get_f1(row)
    if f1 is not None:
        labs.append(lab); vals.append(f1)

if not vals:
    print("[ERR] No matching FILE rows found in variants.csv"); raise SystemExit(1)

fig=plt.figure(figsize=(5,3.2), dpi=120)
ax=fig.add_subplot(111)
x=np.arange(len(vals))
ax.bar(x, vals)
ax.set_xticks(x); ax.set_xticklabels(labs)
ax.set_ylim(0,1); ax.set_ylabel("File F1")
for i,v in enumerate(vals): ax.text(i, v+0.02, f"{v:.3f}", ha="center")
fig.tight_layout()
out = FIGS/"top3_fileF1.png"
fig.savefig(out)
plt.close(fig)
print(f"[OK] saved: {out}")
