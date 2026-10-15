# Best-of table (CSV + PNG): model, preprocess(id), file-F1, (segment-F1 if present), scaler, PCA
from pathlib import Path
import csv, json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ART=HERE/"artifacts"; FIGS=ART/"figs"; FIGS.mkdir(parents=True, exist_ok=True)
vc=ART/"variants.csv"

targets = [
    ("A16-MLP-FILE", "A16-MLP (env+log@8k)"),
    ("A31-MLP-FILE", "A31-MLP (env+log@16k)"),
    ("A8-GMM-FILE",  "A8-GMM (BP1–5k@16k)"),
    ("A42-RF",       "RF(50,depth≤8) on A16"),
    ("A43-ET",       "ET(50,depth≤8) on A16"),
]

def last_row_id(exact):
    if not vc.exists(): return None
    with open(vc,"r",encoding="utf-8") as f:
        rows=list(csv.DictReader(f))
    last=None
    for r in rows:
        rid=(r.get("id") or "").upper()
        if rid==exact.upper(): last=r
    return last

def val_or_blank(r, keys):
    if not r: return ""
    for k in keys:
        if k in r and str(r[k]).strip()!="": return r[k]
    return ""

rows=[]
for rid,label in targets:
    r = last_row_id(rid)
    fileF1 = val_or_blank(r, ["file_F1","file_f1","f1"])
    segF1  = val_or_blank(r, ["segment_F1","seg_f1"])
    scaler = val_or_blank(r, ["scaler"])
    pca    = val_or_blank(r, ["pca","pca_dims"])
    rows.append({
        "label": label,
        "id": rid,
        "file_F1": float(fileF1) if fileF1!="" else np.nan,
        "segment_F1": float(segF1) if segF1!="" else np.nan,
        "scaler": scaler,
        "pca": pca,
        "notes": val_or_blank(r, ["notes"])
    })

# CSV
df=pd.DataFrame(rows)
csv_out = ART/"best_of_table.csv"
df.to_csv(csv_out, index=False, encoding="utf-8")
print(f"[OK] saved: {csv_out}")

# PNG table
fig = plt.figure(figsize=(8, len(rows)*0.7+1), dpi=150)
ax = fig.add_subplot(111); ax.axis("off")
tbl = ax.table(cellText=df.fillna("").values, colLabels=df.columns, loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
tbl.auto_set_column_width(col=list(range(len(df.columns))))
fig.tight_layout()
png_out = FIGS/"best_of_table.png"
fig.savefig(png_out); plt.close(fig)
print(f"[OK] saved: {png_out}")
