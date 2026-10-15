# Check each artifacts/runs/* has code.py, params.json, metrics.json|metrics_file.json, and a cm_*.png
from pathlib import Path
import json, csv

HERE=Path(__file__).resolve().parent
RUNS=(HERE/"artifacts"/"runs")
OUT_CSV = HERE/"artifacts"/"audit_runs_integrity.csv"
RUNS.mkdir(parents=True, exist_ok=True)

rows=[]
for d in sorted(RUNS.glob("*")):
    if not d.is_dir(): continue
    has_code = (d/"code.py").exists()
    has_params = (d/"params.json").exists()
    has_metrics = (d/"metrics.json").exists() or (d/"metrics_file.json").exists()
    has_cm = any(p.name.lower().startswith("cm_") and p.suffix.lower()==".png" for p in d.glob("*.png"))
    rows.append({
        "run_dir": str(d),
        "code.py": has_code, "params.json": has_params,
        "metrics(_file).json": has_metrics, "any_cm_png": has_cm
    })

with open(OUT_CSV,"w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["run_dir"])
    w.writeheader()
    for r in rows: w.writerow(r)

print(f"[OK] saved: {OUT_CSV}")
missing=[r for r in rows if (not r["code.py"] or not r["params.json"] or not r["metrics(_file).json"] or not r["any_cm_png"])]
print(f"[INFO] total runs: {len(rows)}, missing-any: {len(missing)}")
