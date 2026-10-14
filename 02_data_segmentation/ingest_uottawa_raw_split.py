# Ingest raw UOttawa WAVs and split into data/raw/{train,val,test}/{normal,anomaly}
# Source (unchanged): C:\Users\ramin\Desktop\Thesis\UOttawa Bearing Data\uottawa_wav\{normal,fault}\*.wav
# Split: 70% train, 15% val, 15% test (deterministic, RNG=42)

from pathlib import Path
import shutil
import numpy as np

# --- CONFIG ---
SRC = Path(r"C:\Users\ramin\Desktop\Thesis\UOttawa Bearing Data\uottawa_wav")
HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
RATIOS = (0.70, 0.15, 0.15)
RNG = np.random.default_rng(42)

MAP = {
    "normal": "normal",   # src folder -> dest class
    "fault":  "anomaly",
}

def gather_wavs(src_dir: Path):
    return sorted([p for p in src_dir.rglob("*.wav") if p.is_file()])

def split_indices(n, ratios):
    r_train, r_val, r_test = ratios
    i = np.arange(n)
    RNG.shuffle(i)
    n_tr = int(round(n * r_train))
    n_va = int(round(n * r_val))
    n_te = n - n_tr - n_va
    tr = i[:n_tr]
    va = i[n_tr:n_tr+n_va]
    te = i[n_tr+n_va:]
    return tr.tolist(), va.tolist(), te.tolist()

def copy_list(files, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for fp in files:
        dst = out_dir / fp.name
        # if name collides, add an index suffix
        if dst.exists():
            stem, suf = dst.stem, dst.suffix
            k = 1
            while True:
                cand = dst.with_name(f"{stem}__dup{k}{suf}")
                if not cand.exists():
                    dst = cand; break
                k += 1
        shutil.copy2(fp, dst)

def main():
    if not SRC.exists():
        raise SystemExit(f"[ERR] Source not found: {SRC}")

    RAW.mkdir(parents=True, exist_ok=True)

    totals = {}
    for src_name, dest_cls in MAP.items():
        src_cls_dir = SRC / src_name
        if not src_cls_dir.exists():
            print(f"[WARN] Missing: {src_cls_dir} (skipped)")
            continue

        files = gather_wavs(src_cls_dir)
        n = len(files)
        if n == 0:
            print(f"[WARN] No WAVs in {src_cls_dir}")
            continue

        tr_idx, va_idx, te_idx = split_indices(n, RATIOS)
        tr = [files[i] for i in tr_idx]
        va = [files[i] for i in va_idx]
        te = [files[i] for i in te_idx]

        # copy
        copy_list(tr, RAW / "train" / dest_cls)
        copy_list(va, RAW / "val" / dest_cls)
        copy_list(te, RAW / "test" / dest_cls)

        totals[dest_cls] = {
            "train": len(tr),
            "val":   len(va),
            "test":  len(te),
            "total": n,
        }

        print(f"[OK] {dest_cls}: total={n}  -> train={len(tr)}, val={len(va)}, test={len(te)}")

    # Summary + next step hint
    print("\n[SUMMARY]")
    for cls, d in totals.items():
        print(f"  {cls:7s}: {d}")

    print("\n[NEXT]")
    print("Run: resegment_make_B1_B2.py  (will create data/segments_split_B1 and _B2)")

if __name__ == "__main__":
    main()
