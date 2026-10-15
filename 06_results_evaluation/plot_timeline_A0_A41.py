# Timeline A0→A41: F1 per ablation (prefers file_F1, else f1), separate lines for GMM & MLP
from pathlib import Path
import csv, re, numpy as np, matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ART=HERE/"artifacts"; FIGS=ART/"figs"; FIGS.mkdir(parents=True, exist_ok=True)
vc=ART/"variants.csv"

if not vc.exists(): print("[ERR] variants.csv missing"); raise SystemExit(1)
with open(vc,"r",encoding="utf-8") as f:
    rows=list(csv.DictReader(f))

def abla_num(row):
    rid=(row.get("id") or "").upper()
    m=re.match(r"A(\d+)", rid)
    return int(m.group(1)) if m else None

def model_tag(row):
    m=(row.get("model") or "").upper()
    if "GMM" in m: return "GMM"
    if "MLP" in m: return "MLP"
    return None

def f1_val(row):
    for k in ["file_F1","file_f1","f1"]:
        if k in row and row[k]!="":
            try: return float(row[k])
            except: pass
    return None

best={"GMM":{}, "MLP":{}}
for r in rows:
    n=abla_num(r); mt=model_tag(r); f=f1_val(r)
    if n is None or n>41 or mt is None or f is None: continue
    # keep last occurrence (latest appends are last)
    best[mt][n]=f

xs=sorted(set(best["GMM"].keys())|set(best["MLP"].keys()))
if not xs:
    print("[ERR] No A0-A41 rows found"); raise SystemExit(1)

yg=[best["GMM"].get(i, np.nan) for i in xs]
ym=[best["MLP"].get(i, np.nan) for i in xs]

fig=plt.figure(figsize=(7.5,3.2), dpi=120); ax=fig.add_subplot(111)
ax.plot(xs, yg, marker="o", label="GMM")
ax.plot(xs, ym, marker="o", label="MLP")
ax.set_xlabel("Ablation ID"); ax.set_ylabel("F1 (file if available)")
ax.set_xticks(xs); ax.set_xticklabels([f"A{i}" for i in xs], rotation=60)
ax.set_ylim(0,1); ax.legend()
fig.tight_layout()
out=FIGS/"timeline_A0_A41.png"; fig.savefig(out); plt.close(fig)
print(f"[OK] saved: {out}")
