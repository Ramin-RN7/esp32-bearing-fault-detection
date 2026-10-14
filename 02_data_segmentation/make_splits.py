from pathlib import Path
import csv, random, shutil, math
ROOT = Path(__file__).resolve().parent
MANIFESTS = ROOT / "manifests"
SEGROOT = ROOT
IN_MAN = MANIFESTS / "segments.csv"
OUT_ALL = MANIFESTS / "segments_split.csv"
OUT_T = MANIFESTS / "split_train.csv"
OUT_V = MANIFESTS / "split_val.csv"
OUT_E = MANIFESTS / "split_test.csv"

DO_COPY = True
OUTDIR = ROOT / "data" / "segments_split"   # data/segments_split/{train,val,test}/{class}/

RATIO = (0.70, 0.15, 0.15)  # train/val/test
SEED = 42
CLASSES = ["normal","anomaly"]

# --- load rows ---
rows=[]
with open(IN_MAN, newline="") as f:
    r=csv.DictReader(f)
    for row in r:
        rows.append(row)

# --- group by class -> orig_file ---
byfile = {c:{} for c in CLASSES}
for row in rows:
    c = row["class"]
    of = row["orig_file"]
    byfile[c].setdefault(of, []).append(row)

# --- assign splits per orig_file (no leakage) ---
random.Random(SEED)  # init once
assign = {}  # orig_file -> split
for c in CLASSES:
    files = list(byfile[c].keys())
    rnd = random.Random(SEED)  # same seed per class for reproducibility
    rnd.shuffle(files)
    n = len(files)
    n_train = math.floor(RATIO[0]*n)
    n_val = max(1, round(RATIO[1]*n))
    n_test = n - n_train - n_val
    # ensure test >=1 when possible
    if n_test == 0 and n >= 3:
        n_val -= 1; n_test = 1
    train_files = set(files[:n_train])
    val_files   = set(files[n_train:n_train+n_val])
    test_files  = set(files[n_train+n_val:])
    for of in train_files: assign[of] = "train"
    for of in val_files:   assign[of] = "val"
    for of in test_files:  assign[of] = "test"

# --- write outputs and optional copies ---
counts = { "train":{c:0 for c in CLASSES}, "val":{c:0 for c in CLASSES}, "test":{c:0 for c in CLASSES} }
ufiles = { "train":{c:set() for c in CLASSES}, "val":{c:set() for c in CLASSES}, "test":{c:set() for c in CLASSES} }

OUTDIR.mkdir(parents=True, exist_ok=True)
for split in ["train","val","test"]:
    for c in CLASSES:
        (OUTDIR/split/c).mkdir(parents=True, exist_ok=True)

with open(OUT_ALL, "w", newline="") as fa, \
     open(OUT_T, "w", newline="") as ft, \
     open(OUT_V, "w", newline="") as fv, \
     open(OUT_E, "w", newline="") as fe:
    fields = ["filepath","class","orig_file","segment_id","duration_s","split"]
    wa, wt, wv, we = [csv.DictWriter(f, fieldnames=fields) for f in (fa,ft,fv,fe)]
    for w in (wa,wt,wv,we): w.writeheader()

    for row in rows:
        of = row["orig_file"]
        sp = assign[of]
        row_out = dict(row); row_out["split"]=sp
        wa.writerow(row_out)
        if sp=="train": wt.writerow(row_out)
        elif sp=="val": wv.writerow(row_out)
        else: we.writerow(row_out)

        c = row["class"]
        counts[sp][c] += 1
        ufiles[sp][c].add(of)

        if DO_COPY:
            src = (ROOT / row["filepath"]).resolve()
            dst = (OUTDIR/sp/c/f"{Path(row['filepath']).stem}.wav").resolve()
            shutil.copy2(src, dst)

# --- print summary ---
print("== Split summary (segments) ==")
for sp in ["train","val","test"]:
    print(f"{sp}: normal={counts[sp]['normal']}, anomaly={counts[sp]['anomaly']}")
print("\n== Unique original files per split (no leakage check) ==")
for sp in ["train","val","test"]:
    print(f"{sp}: normal_files={len(ufiles[sp]['normal'])}, anomaly_files={len(ufiles[sp]['anomaly'])}")
print(f"\nWrote:\n  {OUT_ALL}\n  {OUT_T}\n  {OUT_V}\n  {OUT_E}\nCopies in: {OUTDIR}")
